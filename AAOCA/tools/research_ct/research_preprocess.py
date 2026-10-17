"""Research CT resampling using official nnU-Net kernels, preserving physical FOV and HU.

This is NOT nnU-Net training planning or model-specific normalization.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import argparse, csv, gc, gzip, hashlib, html, importlib.metadata, inspect, json, os, threading, time
import nibabel as nib
import numpy as np
import psutil
import torch
from scipy.ndimage import map_coordinates
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_simple
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape

from workspace import workspace_root, work_directory
ROOT=workspace_root(__file__)
WORK=work_directory(ROOT)
TARGET=np.array([.5,.5,.5],dtype=float)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU_LOCK=threading.Lock()
LOG_LOCK=threading.Lock()
CONFIG={'schema':1,'purpose':'model_agnostic_research_images','nominal_target_spacing_mm':[.5,.5,.5],
 'orientation':'RAS axis ordering; retain original obliquity',
 'interpolation':'nnunetv2.resample_torch_simple, linear (3D trilinear; large case separable bilinear passes)',
 'grid':'shape rounded by nnU-Net; affine tracks actual N/M sampling with align_corners=False, preserving voxel-cell FOV and centre',
 'intensity':'HU; no clipping, no windowing, no intensity normalization','crop':'none','dtype':'float32',
 'nnunetv2':importlib.metadata.version('nnunetv2'),'torch':torch.__version__,
 'kernel_file_sha256':hashlib.sha256(Path(inspect.getfile(resample_torch_simple)).read_bytes()).hexdigest()}
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def write_json(path,v):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def grid(shape,affine):
 spacing=nib.affines.voxel_sizes(affine)
 size=compute_new_shape(shape,spacing,TARGET).astype(int)
 assert (size>0).all()
 ratio=np.asarray(shape,dtype=float)/size
 transform=np.eye(4);transform[:3,:3]=np.diag(ratio);transform[:3,3]=(ratio-1)/2
 new_affine=affine@transform
 # Check both boundary corners and centre: all use voxel-cell, not centre-to-centre extent.
 for a,b in [(-np.ones(3)*.5,-np.ones(3)*.5),(np.asarray(shape)-.5,size-.5),((np.asarray(shape)-1)/2,(size-1)/2)]:
  assert np.allclose(affine@np.r_[a,1],new_affine@np.r_[b,1],atol=1e-8)
 return size,new_affine,transform

def kernel(data,shape):
 with GPU_LOCK:
  return np.asarray(resample_torch_simple(np.ascontiguousarray(data,dtype=np.float32),tuple(map(int,shape)),is_seg=False,num_threads=4,device=DEVICE,mode='linear'),dtype=np.float32)

def separable(data,new_shape,tag):
 """Exact linear separability keeps the 12 GiB case bounded without shrinking FOV."""
 nx,ny,nz=map(int,new_shape);oz=data.shape[2]
 scratch=WORK/'scratch';scratch.mkdir(exist_ok=True)
 midpath=scratch/f'{tag}.xy.float32';outpath=scratch/f'{tag}.out.float32'
 mid=np.memmap(midpath,dtype=np.float32,mode='w+',shape=(oz,nx,ny),order='C')
 for z0 in range(0,oz,16):
  z1=min(oz,z0+16)
  batch=np.transpose(data[:,:,z0:z1],(2,0,1))
  mid[z0:z1]=kernel(batch,(nx,ny))
  if z0%160==0:print(f'CHUNK {tag} XY {z0}/{oz}',flush=True)
 mid.flush()
 output=np.memmap(outpath,dtype=np.float32,mode='w+',shape=(nx,ny,nz),order='F')
 for x0 in range(0,nx,24):
  x1=min(nx,x0+24)
  batch=np.asarray(mid[:,x0:x1,:]).transpose(1,2,0).copy().reshape((x1-x0)*ny,1,oz)
  result=kernel(batch,(1,nz)).reshape(x1-x0,ny,nz)
  output[x0:x1]=result
  del batch,result
  if x0%120==0:print(f'CHUNK {tag} Z {x0}/{nx}',flush=True)
 output.flush();del mid;gc.collect();midpath.unlink()
 return output,outpath

def statistics(a):
 count=0;lo=float('inf');hi=-float('inf');total=0.;squares=0.
 for k in range(0,a.shape[2],8):
  slab=np.asarray(a[:,:,k:k+8])
  if not np.isfinite(slab).all():raise ValueError('像素包含 NaN 或无穷值')
  count+=slab.size;lo=min(lo,float(slab.min()));hi=max(hi,float(slab.max()))
  total+=float(slab.sum(dtype=np.float64));squares+=float(np.square(slab,dtype=np.float64).sum())
 mean=total/count
 return {'min':lo,'max':hi,'mean':mean,'std':float(np.sqrt(max(0,squares/count-mean**2))),'finite':True,'voxels':count}

def payload_hash(a):
 h=hashlib.sha256()
 # Each block is serialized in the same X-fastest order as NIfTI.
 for z in range(0,a.shape[2],8):h.update(np.asarray(a[:,:,z:z+8],dtype='<f4').tobytes(order='F'))
 return h.hexdigest()

def validate_written(path,array,affine):
 image=nib.load(str(path))
 assert image.shape==array.shape and image.get_data_dtype()==np.dtype('float32')
 assert np.allclose(image.affine,affine,atol=1e-4)
 assert nib.aff2axcodes(image.affine)==('R','A','S')
 assert image.header.get_xyzt_units()[0]=='mm'
 assert image.dataobj.slope==1 and image.dataobj.inter==0
 expected=payload_hash(array);h=hashlib.sha256();n=0
 with gzip.open(path,'rb') as f:
  prefix=f.read(int(image.dataobj.offset));assert len(prefix)==image.dataobj.offset
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk);n+=len(chunk)
 assert n==array.size*4 and h.hexdigest()==expected,'保存后像素与内存数组不一致'
 return {'full_gzip_read_and_crc':True,'voxel_payload_sha256':expected,'all_voxels_roundtrip_equal':True,
         'nifti_affine_max_error_mm':float(np.abs(image.affine-affine).max())}

def sampling_check(original,output,transform):
 rng=np.random.default_rng(20261017)
 positions=np.column_stack([rng.integers(0,n,size=512) for n in output.shape]).T
 mapped=(transform@np.vstack([positions,np.ones(positions.shape[1])]))[:3]
 expected=map_coordinates(original,mapped,order=1,mode='nearest',prefilter=False)
 actual=output[tuple(positions)]
 delta=np.abs(expected.astype(float)-actual.astype(float))
 assert delta.max()<1.0,f'独立线性插值复核差异过大: {delta.max()} HU'
 return {'sample_count':512,'independent_scipy_linear_max_difference_HU':float(delta.max()),'mean_difference_HU':float(delta.mean())}

def make_preview(original,output,spacing,new_spacing,path):
 from PIL import Image,ImageDraw
 canvas=Image.new('RGB',(1000,830),'#f3f6fa');draw=ImageDraw.Draw(canvas)
 draw.text((15,10),'Native RAS (left) / research grid (right); display window -200..800 HU',fill='black')
 for row,axis in enumerate([2,1,0]):
  for col,(a,sp) in enumerate([(original,spacing),(output,new_spacing)]):
   im=np.take(a,a.shape[axis]//2,axis=axis).T
   remaining=[i for i in range(3) if i!=axis]
   width=a.shape[remaining[0]]*sp[remaining[0]];height=a.shape[remaining[1]]*sp[remaining[1]]
   pixels=np.uint8(np.clip((im+200)/1000,0,1)*255)
   thumbnail=Image.fromarray(pixels)
   factor=min(460/width,235/height)
   thumbnail=thumbnail.resize((max(1,round(width*factor)),max(1,round(height*factor))),Image.Resampling.BILINEAR)
   canvas.paste(thumbnail,(col*500+(500-thumbnail.width)//2,50+row*260+(235-thumbnail.height)//2))
  draw.text((15,35+row*260),['Axial','Coronal','Sagittal'][row],fill='black')
 canvas.save(path)

def process(case,preview=False):
 start=time.monotonic();src=ROOT/case['destination']
 directory=src.parent/'研究版_目标0.5mm';dst=directory/src.name;recordpath=directory/'预处理记录.json'
 if recordpath.exists() and dst.exists():
  old=json.loads(recordpath.read_text(encoding='utf-8'))
  assert old['config_sha256']==CONFIG_HASH and old['input_sha256']==case['sha256'],'现有研究版使用不同参数或源文件'
  assert sha(dst)==old['output_sha256'],'已有研究版文件校验失败'
  return {**old,'reused':True}
 assert sha(src)==case['sha256'],'输入 CT 校验值改变'
 im=nib.load(str(src));canonical=nib.as_closest_canonical(im)
 assert nib.aff2axcodes(canonical.affine)==('R','A','S')
 assert im.header.get_xyzt_units()[0]=='mm'
 data=np.asanyarray(canonical.dataobj,dtype=np.float32)
 input_stats=statistics(data)
 size,affine,mapping=grid(data.shape,canonical.affine)
 expected_bytes=int(np.prod(size))*4
 temporary=None
 if expected_bytes>2*2**30:
  output,temporary=separable(data,size,case['case_id']);method='separable_nnUNet_bilinear_xy_then_z'
 else:
  output=kernel(data[None],size)[0];method='nnUNet_trilinear'
 verification=sampling_check(data,output,mapping)
 stats=statistics(output)
 assert stats['min']>=input_stats['min']-.1 and stats['max']<=input_stats['max']+.1
 directory.mkdir(parents=True,exist_ok=True)
 partial=directory/(case['case_id']+'.partial.nii.gz')
 img=nib.Nifti1Image(output,affine)
 img.set_sform(affine,code=1);img.set_qform(affine,code=1)
 img.header.set_xyzt_units('mm');img.set_data_dtype(np.float32)
 nib.save(img,str(partial))
 verification.update(validate_written(partial,output,affine))
 record={'case_id':case['case_id'],'type':case['type'],'number':case['number'],'registered_patient_id':case['registered_patient_id'],
  'input':str(src.relative_to(ROOT)),'output':str(dst.relative_to(ROOT)),
  'input_sha256':case['sha256'],'output_sha256':sha(partial),'config_sha256':CONFIG_HASH,'config':CONFIG,
  'original_shape':list(im.shape),'original_affine':im.affine.tolist(),
  'canonical_shape':list(data.shape),'canonical_affine':canonical.affine.tolist(),
  'output_shape':list(map(int,size)),'output_affine':affine.tolist(),
  'nominal_spacing_mm':TARGET.tolist(),'actual_spacing_mm':nib.affines.voxel_sizes(affine).tolist(),
  'output_index_to_canonical_index':mapping.tolist(),
  'output_index_to_original_index':(np.linalg.inv(im.affine)@affine).tolist(),
  'original_index_to_output_index':(np.linalg.inv(affine)@im.affine).tolist(),
  'input_statistics_HU':input_stats,'output_statistics_HU':stats,'verification':verification,
  'method':method,'completed_at':now(),'seconds':time.monotonic()-start,'device':str(DEVICE)}
 if preview:make_preview(data,output,nib.affines.voxel_sizes(canonical.affine),nib.affines.voxel_sizes(affine),directory/'处理前后对照.png')
 partial.replace(dst);write_json(recordpath,record)
 del img,output,data,canonical,im;gc.collect()
 if temporary is not None:temporary.unlink()
 with GPU_LOCK:
  if DEVICE.type=='cuda':torch.cuda.empty_cache()
 return record

def verify_math():
 x,y,z=np.indices((11,13,7));data=(20*x+3*y-5*z+np.sin(x+y)).astype(np.float32)
 affine=np.array([[-.73,0,0,82],[0,-.59,0,35],[0,0,1.7,-10],[0,0,0,1]],dtype=float)
 original=nib.Nifti1Image(data,affine);can=nib.as_closest_canonical(original)
 a=np.asanyarray(can.dataobj);shape,out_aff,t=grid(a.shape,can.affine)
 full=kernel(a[None],shape)[0];chunk,tmp=separable(a,shape,'synthetic_geometry_check')
 difference=float(np.abs(full-chunk).max());assert difference<.001
 checks=sampling_check(a,full,t)
 # world coordinates mapped back to the original, asymmetric phantom.
 p=np.array([3,4,2,1.]);old=np.linalg.inv(affine)@out_aff@p
 expected=map_coordinates(data,old[:3,None],order=1,mode='nearest',prefilter=False)[0]
 assert abs(float(full[3,4,2])-float(expected))<.001
 del chunk;gc.collect();tmp.unlink()
 write_json(WORK/'geometry_kernel_verification.json',{'time':now(),'config':CONFIG,'chunk_vs_full_max_error_HU':difference,'independent_samples':checks,'asymmetric_orientation_and_world_coordinate_test':True})
 print('SYNTHETIC GEOMETRY VERIFIED',difference,flush=True)

def progress(done,errors,total,running):
 info={'updated':now(),'stage':'处理中' if len(done)+len(errors)<total else ('完成' if not errors else '完成，部分待处理'),
       'total':total,'success':len(done),'failed':len(errors),'running':running,'errors':errors,'config':CONFIG}
 write_json(ROOT/'预处理进度.json',info)
 page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="15"><title>先心 CT 研究版处理进度</title><style>body{font-family:Microsoft YaHei,sans-serif;max-width:1000px;margin:45px auto;line-height:1.8}b{color:#135c9d}pre{white-space:pre-wrap}</style>'
 page+=f'<h1>先心 CT 研究版处理进度</h1><p>{html.escape(info["updated"])} · {info["stage"]}</p><h2>已完成 <b>{len(done)} / {total}</b> · 未成功 {len(errors)}</h2><p>目标近 0.5 mm，RAS 轴顺序，保留 HU 和完整扫描范围。使用 nnU-Net 重采样组件；未进行模型专属归一化。</p><pre>'+html.escape('\n'.join(running))+'</pre><p>每份输出完成后校验全部像素回读、空间信息和原图映射，并保存预处理记录。</p>'
 if errors:page+='<pre>'+html.escape(json.dumps(errors,ensure_ascii=False,indent=2))+'</pre>'
 tmp=ROOT/'预处理进度.html.tmp';tmp.write_text(page,encoding='utf-8');tmp.replace(ROOT/'预处理进度.html')

def run(pilot=False):
 cases=json.loads((ROOT/'CT总清单.json').read_text(encoding='utf-8'))
 if pilot:
  checkpoint=WORK/'pilot_results.json'
  previous=json.loads(checkpoint.read_text(encoding='utf-8')).get('completed',[]) if checkpoint.exists() else []
  previous_ids={r['case_id'] for r in previous}
  selected=[c for c in cases if c['case_id'] in previous_ids]
  if not selected:
   selected=[next(c for c in cases if c['type']==kind) for kind in sorted({c['type'] for c in cases})]
   selected.extend(c for c in cases if c not in selected)
  cases=selected[:3]
 done=[];errors=[];running=[]
 write_json(ROOT/'研究版处理方案.json',{'created':now(),'config':CONFIG,'config_sha256':CONFIG_HASH,'user_intent':'统一研究用影像，模型以后再定；跳过加密 CHD 标签核对'})
 def execute(c):
  with LOG_LOCK:running.append(c['case_id']);progress(done,errors,len(cases),running)
  try:
   result=process(c,preview=pilot or int(np.prod(compute_new_shape(c['shape'],c['spacing_mm'],TARGET)))*4>2*2**30)
   with LOG_LOCK:
    done.append(result)
    with (WORK/'preprocess_completed.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
  except Exception as ex:
   import traceback
   with LOG_LOCK:errors.append({'case_id':c['case_id'],'error':str(ex),'traceback':traceback.format_exc()})
  finally:
   with LOG_LOCK:
    running.remove(c['case_id']);progress(done,errors,len(cases),running)
    print(f'PREPROCESS {len(done)+len(errors)}/{len(cases)} success={len(done)} failed={len(errors)} {c["case_id"]}',flush=True)
 large=[c for c in cases if np.prod(compute_new_shape(c['shape'],c['spacing_mm'],TARGET))*4>2*2**30]
 normal=[c for c in cases if c not in large]
 with ThreadPoolExecutor(max_workers=1 if pilot else 2) as pool:
  futures=[pool.submit(execute,c) for c in normal]
  for f in as_completed(futures):f.result()
 for c in large:execute(c)
 output=WORK/'pilot_results.json' if pilot else ROOT/'预处理清单.json'
 write_json(output,{'completed':done,'errors':errors,'config':CONFIG})
 if not pilot:
  with (ROOT/'预处理清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
   fields=['type','number','registered_patient_id','case_id','input','output','original_shape','output_shape','actual_spacing_mm','output_sha256','method','seconds']
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
   for r in sorted(done,key=lambda x:x['case_id']):w.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],list) else r[k] for k in fields})
 print('FINISHED',len(done),'ERRORS',errors,flush=True)
 if errors:raise SystemExit(1)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['verify','pilot','all']);args=p.parse_args()
 torch.set_num_threads(4)
 {'verify':verify_math,'pilot':lambda:run(True),'all':run}[args.action]()
