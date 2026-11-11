"""Derived supervision in prepared-cache coordinates. Published outputs contain no original CT."""
from pathlib import Path
import os
import tempfile

import numpy as np
from scipy import ndimage

from .contracts import atomic_json, digest, read_json, sha256_file
from .xu_geometry import _box_in, _resize

SEMANTICS = dict(LV='left_ventricle', RV='right_ventricle', LA='left_atrium',
                 RA='right_atrium', MYO='myocardium', AO='aorta', PA='pulmonary_artery',
                 PV='pulmonary_veins', SVC='superior_vena_cava', IVC='inferior_vena_cava')
PROTOCOL = 'xu-aux-targets-v1'


def anatomy(cache):
    """Use the verified provider/local label dictionary, never infer names from integer order."""
    labels = read_json(cache.run_dir / 'preprocessed/Dataset282_CHDSource/dataset.json')['labels']
    if (set(labels) != {'background', *SEMANTICS.values()} or labels['background'] != 0
            or any(type(v) is not int for v in labels.values()) or set(labels.values()) != set(range(11))):
        raise ValueError('A complete, unambiguous 10-structure dictionary is required')
    return {k: labels[v] for k, v in SEMANTICS.items()}


def blood_ids(cache):
    return tuple(v for k, v in anatomy(cache).items() if k != 'MYO')


def blood_eligible(supervision, ids):
    observed = set(supervision['present_classes']) | set(supervision['known_absent_classes'])
    return set(ids) <= observed and not set(supervision['unknown_classes']).intersection(ids)


def _spacing(spacing, dimensions):
    step = np.asarray(spacing, dtype=float)
    if step.shape != (dimensions,) or not np.all(np.isfinite(step) & (step > 0)):
        raise ValueError('Expected finite positive spacing in array axis order')
    return step


def _label_grid(seg):
    seg = np.asarray(seg)
    if seg.ndim not in (2, 3) or not seg.size or seg.dtype.kind not in 'iu' or seg.min() < -1 or seg.max() > 10:
        raise ValueError('Invalid prepared anatomical label array')
    return seg


def blood_target(seg, spacing, ids, width_mm=1.0, interface=None):
    """Inner boundary band by centre distance to observed non-blood in millimetres.

    Distances to padding and to unobserved tissue do not create a wall. A partial
    band bordering ignored voxels remains unknown, except for observed walls.
    """
    seg = _label_grid(seg)
    step = _spacing(spacing, seg.ndim)
    if not np.isfinite(width_mm) or width_mm <= 0:
        raise ValueError('Boundary width must be positive and finite')
    lumen = np.isin(seg, ids)
    unknown = seg == -1
    nonblood = ~(lumen | unknown)
    out = lumen.astype(np.int8)
    # Slabs with finite halos compute exactly the same finite band as a full
    # Euclidean transform. Avoid allocating an EDT over a whole cardiac scan.
    halo = min(seg.shape[0], int(np.ceil(width_mm / step[0])) + 1)
    for lo in range(0, seg.shape[0], 12):
        hi = min(seg.shape[0], lo + 12)
        a, b = max(0, lo-halo), min(seg.shape[0], hi+halo)
        core = slice(lo-a, hi-a)
        if unknown[a:b].any() and lumen[lo:hi].any():
            dist = ndimage.distance_transform_edt(~unknown[a:b], sampling=step)[core]
            out[lo:hi][lumen[lo:hi] & (dist <= width_mm)] = -1
        if nonblood[a:b].any() and lumen[lo:hi].any():
            dist = ndimage.distance_transform_edt(~nonblood[a:b], sampling=step)[core]
            out[lo:hi][lumen[lo:hi] & (dist <= width_mm)] = 2
    out[unknown] = -1
    if interface is not None:
        interface = np.asarray(interface)
        if interface.shape != seg.shape or interface.dtype != np.bool_ or np.any(interface & ~lumen):
            raise ValueError('Reviewed interfaces must lie on known blood-pool voxels')
        out[interface] = 2
    return out


def cache_coordinates(geometry):
    """Physical coordinates of prepared cell centres along each transposed axis."""
    axes = [2-i for i in geometry.transpose_forward]
    origin, step = [], []
    for a, (start, stop), n in zip(axes, geometry.bbox_transposed, geometry.cache_shape):
        s = geometry.spacing_xyz[a]
        size = (stop-start)*s/n
        origin.append((start-.5)*s+.5*size)
        step.append(size)
    return axes, np.array(origin), np.array(step)


def _vector(row):
    v = np.asarray(row, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('Expected a finite 3-vector in local XYZ millimetres')
    return v


def _proximal_component(seg, vessel, axes, origin, step, spec):
    """Clip a reviewed vessel component between two oriented cut planes."""
    mask = seg == vessel
    if not mask.any():
        raise ValueError('Reviewed proximal vessel absent in prepared label')
    if set(spec) != {'start', 'end', 'seed_xyz_mm'}:
        raise ValueError('Each vessel review needs start/end planes and an interior seed')
    start_point = _vector(spec['start']['point_xyz_mm'])
    end_point = _vector(spec['end']['point_xyz_mm'])
    if np.linalg.norm(end_point - start_point) < min(step):
        raise ValueError('Review start and end planes are indistinguishable at this resolution')
    seed = _vector(spec['seed_xyz_mm'])[axes]
    centre = np.floor((seed-origin)/step+.5).astype(int)
    if np.any(centre < 0) or np.any(centre >= seg.shape) or not mask[tuple(centre)]:
        raise ValueError('Reviewed seed falls outside vessel')
    region = ndimage.find_objects(mask, max_label=1)[0]
    cropped = mask[region]
    coords = [origin[a] + np.arange(s.start, s.stop)*step[a] for a, s in enumerate(region)]
    for key, sign in [('start',1), ('end',-1)]:
        plane = spec[key]
        if set(plane) != {'point_xyz_mm','normal_xyz'}:
            raise ValueError('Expected point_xyz_mm and downstream normal_xyz')
        point = _vector(plane['point_xyz_mm'])[axes]
        normal = _vector(plane['normal_xyz'])[axes]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            raise ValueError('Degenerate cut plane')
        normed = normal / norm
        signed = sum(x.reshape(tuple(-1 if j==a else 1 for j in range(3))) * normed[a]
                     for a,x in enumerate(coords)) - np.dot(point,normed)
        if np.min(np.abs(signed[mask[region]])) > np.linalg.norm(step):
            raise ValueError('Cut plane does not intersect the reviewed vessel')
        cropped = cropped & (signed * sign >= -1e-7)
    labels, _ = ndimage.label(cropped)
    selected = labels[tuple(centre - [sl.start for sl in region])]
    if selected == 0:
        raise ValueError('Planes exclude the reviewed seed')
    result = np.zeros(mask.shape, bool)
    result[region] = labels == selected
    return result


def proximal_target(seg, geometry, mapping, review):
    seg = _label_grid(seg)
    if seg.shape != tuple(geometry.cache_shape):
        raise ValueError('Reviewed planes apply only to the specified prepared grid')
    if (review.get('approved') is not True or not isinstance(review.get('review_id'), str)
            or not review['review_id'].strip()):
        raise ValueError('Explicit reviewed planes and review ID required')
    axes, origin, step = cache_coordinates(geometry)
    target = np.zeros(seg.shape, np.int8)
    target[seg == -1] = -1
    for idx, key in enumerate(('LV','RV','LA','RA','MYO'), 1):
        target[seg == mapping[key]] = idx
    for key in ('AO','PA'):
        target[_proximal_component(seg, mapping[key], axes, origin, step, review[key])] = 6
    return target


class TargetStore:
    def __init__(self, root, cache):
        from .xu_workflow import artifact_identity
        self.root = Path(root).resolve()
        self.cache = cache
        self.manifest = read_json(self.root/'targets.json')
        c = self.manifest['contract']
        from .xu_auto_init import PROTOCOL as AUTO_PROTOCOL, validate_auto_contract
        if (c['protocol'] not in (PROTOCOL, AUTO_PROTOCOL) or c['request'] != cache.request_digest
                or c != read_json(self.root/'target-request.json')
                or set(self.manifest['cases']) != set(cache.case_ids('train')) | set(cache.case_ids('development'))
                or c['artifacts'] != artifact_identity(cache) or c['anatomy'] != anatomy(cache)):
            raise ValueError('Auxiliary targets from another preparation')
        if c['protocol'] == AUTO_PROTOCOL:
            validate_auto_contract(c, cache)
            if any(set(row['available']) - {'init'} for row in self.manifest['cases'].values()):
                raise ValueError('Automatic proximal books provide init supervision only')
        self.identity = digest(self.manifest)
        self._verified = set()

    def has(self, name, kind):
        if kind not in ('blood','init'):
            raise ValueError('Unknown auxiliary target')
        return kind in self.manifest['cases'][name]['available']

    def read(self, name, kind, box=None, shape=None):
        import blosc2
        if not self.has(name, kind):
            raise ValueError('No eligible '+kind+' supervision for this case')
        record = self.manifest['cases'][name]['available'][kind]
        path = self.root/'cases'/name/(kind+'.b2nd')
        if path.resolve() != path or not path.is_file():
            raise ValueError('Missing derived target')
        st = path.stat()
        stamp = (name, kind, st.st_size, st.st_mtime_ns)
        if stamp not in self._verified:
            marker = read_json(path.parent/'targets.json')
            if (marker['contract'] != digest(self.manifest['contract'])
                    or marker['result'] != self.manifest['cases'][name] or sha256_file(path) != record):
                raise ValueError('Derived label digest mismatch')
            self._verified.add(stamp)
        arr = blosc2.open(str(path),mode='r')
        expected = self.cache.geometry(name).cache_shape
        if tuple(arr.shape) != tuple(expected) or arr.dtype != np.int8:
            raise ValueError('Invalid auxiliary target grid')
        volume = arr[tuple(slice(a,b) for a,b in _box_in(box,expected))]
        return _resize(volume,shape,0)


def _write_array(path, values):
    import blosc2
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        staged = Path(tmp)/path.name
        array = blosc2.asarray(values, urlpath=str(staged), mode='w')
        del array
        os.replace(staged,path)
    return sha256_file(path)


def _case_targets(cache, out, name, mapping, ids, width_mm, review, review_dir):
    meta = cache.supervision[name]
    available, reasons = {}, {}
    blood_ok = blood_eligible(meta,ids)
    proximal_ids = {mapping[k] for k in ('AO','PA','MYO','LA','RA','LV','RV')}
    init_ok = (review.get('approved') is True and {'AO','PA'} <= set(review)
               and proximal_ids <= set(meta['present_classes']))
    if blood_ok or init_ok:
        seg = cache.read_label(name)
        geometry = cache.geometry(name)
        parent = out/'cases'/name
        if blood_ok:
            interface = None
            if 'interface' in review:
                if review.get('approved') is not True or not str(review.get('review_id','')).strip():
                    raise ValueError('A reviewed inner interface needs explicit approval and ID')
                entry = review['interface']
                path = (review_dir/entry['file']).resolve()
                if not path.is_relative_to(review_dir) or path.suffix != '.npy' or sha256_file(path) != entry['sha256']:
                    raise ValueError('Invalid reviewed interface path or hash')
                interface = np.load(path, allow_pickle=False)
            _,_,spacing = cache_coordinates(geometry)
            blood = blood_target(seg,spacing,ids,width_mm,interface)
            available['blood'] = _write_array(parent/'blood.b2nd',blood)
            del blood
        if init_ok:
            init = proximal_target(seg,geometry,mapping,review)
            available['init'] = _write_array(parent/'init.b2nd',init)
    if not blood_ok:
        reasons['blood'] = 'unverified missing blood component'
    if not init_ok:
        reasons['init'] = 'missing reviewed plane pair or incomplete proximal/chamber supervision'
    return dict(available=available, skipped=reasons)


def build_targets(cache, args, out):
    from .xu_workflow import artifact_identity
    out = Path(out).resolve()
    mapping = anatomy(cache)
    names = [*cache.case_ids('train'), *cache.case_ids('development')]
    reviews = {}
    review_dir = None
    if args.reviews:
        review_dir = args.reviews.resolve().parent
        payload = read_json(args.reviews)
        if (payload.get('request_digest') != cache.request_digest
                or payload.get('coordinate_frame') != 'local_array_excel_mm'
                or set(payload['cases']) - set(names)):
            raise ValueError('Review book geometry or cohort differs')
        reviews = payload['cases']
    width = args.width_mm
    if not np.isfinite(width) or width <= 0:
        raise ValueError('Invalid boundary width')
    contract = dict(protocol=PROTOCOL, request=cache.request_digest, artifacts=artifact_identity(cache),
                    code=sha256_file(Path(__file__)), anatomy=mapping, width_mm=width, reviews=digest(reviews))
    manifest = out/'targets.json'
    request = out/'target-request.json'
    if (request.exists() != args.resume or (args.resume and read_json(request) != contract)
            or (not args.resume and (manifest.exists() or (out/'cases').exists()))):
        raise ValueError('Use a fresh directory or --resume with identical target inputs')
    atomic_json(request, contract)
    ids = blood_ids(cache)
    cases = {}
    for i,name in enumerate(names,1):
        marker = out/'cases'/name/'targets.json'
        if args.resume and marker.exists():
            saved = read_json(marker)
            if saved['contract'] != digest(contract):
                raise ValueError('Target marker contract changed')
            record = saved['result']
            for kind,h in record['available'].items():
                if kind not in ('init','blood') or sha256_file(marker.parent/(kind+'.b2nd')) != h:
                    raise ValueError('Committed auxiliary target differs')
        else:
            record = _case_targets(cache,out,name,mapping,ids,width,reviews.get(name,{}),review_dir)
            atomic_json(marker, dict(contract=digest(contract), result=record))
        cases[name] = record
        print(f'Targets {i}/{len(names)}: {sorted(record["available"])}; skipped {record["skipped"]}',flush=True)
    atomic_json(manifest,dict(contract=contract,cases=cases))
    return dict(command='build-targets',cases=len(cases),manifest=str(manifest),
                available={t:sum(t in r['available'] for r in cases.values()) for t in ('blood','init')})


def review_views(cache, args, out):
    """Write local-coordinate review slices and an explicitly unapproved plane-book template."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    name = args.case
    if name not in cache.supervision:
        raise ValueError('Select a case from the prepared cache')
    target = out/'review-template.json'
    figure_file = out/'review.png'
    if target.exists() or figure_file.exists():
        raise ValueError('Use a fresh review-view directory')
    geo = cache.geometry(name)
    axes, origin, step = cache_coordinates(geo)
    shape = np.array(geo.cache_shape)
    mapping = anatomy(cache)
    if args.center_xyz_mm is None:
        # A scout selects the display location only. It never supplies a cut plane.
        scout = cache.read_label(name, (64,)*3)
        vessels = (scout==mapping['AO']) | (scout==mapping['PA'])
        centre = ((np.mean(np.nonzero(vessels),axis=1)+.5)*shape/64-.5
                  if vessels.any() else (shape-1)/2)
    else:
        centre = (_vector(args.center_xyz_mm)[axes]-origin)/step
    index = np.floor(centre+.5).astype(int)
    if np.any(index < 0) or np.any(index >= shape):
        raise ValueError('Requested local-mm centre outside the cache grid')
    where = origin+index*step
    figure = Figure(figsize=(14,5),layout='constrained')
    FigureCanvasAgg(figure)
    panels = figure.subplots(1,3)
    for panel, (x,y,fixed) in zip(panels,[(0,1,2),(0,2,1),(1,2,0)]):
        ax = axes.index(fixed)
        box = [(0,int(n)) for n in shape]
        box[ax] = (int(index[ax]),int(index[ax])+1)
        remaining = [i for i in axes if i!=fixed]
        order = (remaining.index(y),remaining.index(x))
        ct = cache.read_image(name,box=box).squeeze(ax).transpose(order)
        label = cache.read_label(name,box=box).squeeze(ax).transpose(order)
        extent=[]
        for dim in (x,y):
            a = axes.index(dim)
            edge = origin[a]-step[a]/2
            extent.extend((edge, edge+shape[a]*step[a]))
        lo,hi = np.percentile(ct,[1,99.5])
        panel.imshow(ct,extent=extent,origin='lower',cmap='gray',vmin=lo,vmax=max(hi,lo+1e-6))
        for key, color in [('AO','gold'), ('PA','cyan')]:
            mask = label==mapping[key]
            if mask.any() and not mask.all():
                panel.contour(mask.astype(float),levels=[.5],colors=color,origin='lower',extent=extent,linewidths=.7)
        panel.set(xlabel=f'Local {"XYZ"[x]} (mm)',ylabel=f'Local {"XYZ"[y]} (mm)',
                  title=f'{"XYZ"[fixed]} = {where[ax]:.2f} mm')
    figure.suptitle('Prepared CT  |  AO: gold, PA: cyan  |  local array coordinates')
    figure.savefig(figure_file,dpi=115)
    plane = dict(point_xyz_mm=None,normal_xyz=None)
    vessel = dict(start=plane,end=plane,seed_xyz_mm=None)
    case = dict(approved=False,review_id='',AO=vessel,PA=vessel)
    atomic_json(target,dict(coordinate_frame='local_array_excel_mm',
                            request_digest=cache.request_digest,cases={name:case}))
    return dict(command='review-targets',review_template=str(target),figure=str(figure_file))


def register(subparsers):
    for name in ('review-targets','build-targets','build-auto-targets','train-init'):
        p = subparsers.add_parser(name)
        p.add_argument('--prepared',type=Path,required=True)
        p.add_argument('--technical-subset',action='store_true')
        p.add_argument('--output',type=Path,required=True)
        if name == 'review-targets':
            p.add_argument('--case',required=True)
            p.add_argument('--center-xyz-mm',nargs=3,type=float)
            continue
        p.add_argument('--resume',action='store_true')
        if name=='build-auto-targets':
            p.add_argument('--radius-mm',type=float,default=30.0)
            p.add_argument('--margin-percent',type=float,default=20.0)
        elif name=='build-targets':
            p.add_argument('--reviews',type=Path,help='Private, reviewed local-mm cut planes/interfaces')
            p.add_argument('--width-mm',type=float,default=1.0)
        else:
            p.add_argument('--targets',type=Path,required=True)
            p.add_argument('--boxes',type=Path,required=True)
            p.add_argument('--branch',choices=('init64','init128'),required=True)
            p.add_argument('--profile',choices=('smoke','paper'),default='paper')
            p.add_argument('--supervision',choices=('complete','marginal'),default='complete')
            p.add_argument('--seed',type=int,default=2023)
            p.add_argument('--device',default='cpu')


def run(cache, args, out):
    if args.command=='build-auto-targets':
        from .xu_auto_init import build_auto_targets
        return build_auto_targets(cache,args,out)
    if args.command=='review-targets':
        return review_views(cache,args,out)
    if args.command=='build-targets':
        return build_targets(cache,args,out)
    from .xu_init import train_init
    return train_init(cache,args,out)
