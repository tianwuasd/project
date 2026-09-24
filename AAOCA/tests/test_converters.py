import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, UltrasoundImageStorage, UltrasoundMultiFrameImageStorage, generate_uid

from common import prepare_output, ConversionError
from convert_ct import convert_series
from convert_us import convert_file, frame_timing


def dicom(path, pixels, modality='CT', position=(0,0,0), orientation=(1,0,0,0,1,0), study=None, series=None, frames=None):
    meta=FileMetaDataset()
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=CTImageStorage if modality=='CT' else (UltrasoundMultiFrameImageStorage if frames is not None else UltrasoundImageStorage)
    meta.MediaStorageSOPInstanceUID=generate_uid()
    d=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    d.SOPClassUID=meta.MediaStorageSOPClassUID;d.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
    d.StudyInstanceUID=study or generate_uid();d.SeriesInstanceUID=series or generate_uid()
    d.PatientID='TEST_ONLY';d.Modality=modality
    d.Rows=pixels.shape[-2];d.Columns=pixels.shape[-1]
    d.SamplesPerPixel=1;d.PhotometricInterpretation='MONOCHROME2'
    d.BitsAllocated=pixels.dtype.itemsize*8;d.BitsStored=d.BitsAllocated;d.HighBit=d.BitsStored-1
    d.PixelRepresentation=int(pixels.dtype.kind=='i')
    d.PixelData=pixels.tobytes()
    if frames is not None:d.NumberOfFrames=frames
    if modality=='CT':
        d.ImagePositionPatient=list(position);d.ImageOrientationPatient=list(orientation)
        d.PixelSpacing=[2,3];d.SliceThickness=1
        d.RescaleSlope=2;d.RescaleIntercept=-1000
    d.save_as(path,enforce_file_format=True)
    return d


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.study=generate_uid();self.series=generate_uid()
    def tearDown(self):self.tmp.cleanup()
    def ct(self, name, pos, value=1, orientation=(1,0,0,0,1,0)):
        path=self.root/name
        dicom(path,np.full((2,3),value,dtype=np.int16),position=pos,orientation=orientation,study=self.study,series=self.series)
        return path

    def test_ct_sort_hu_and_oblique_coordinates(self):
        orientation=(0,1,0,0,0,1)  # normal = +X
        p1=self.ct('a.dcm',(12,20,30),3,orientation)
        p0=self.ct('z.dcm',(10,20,30),1,orientation)
        meta=convert_series([p1,p0],self.root/'ct_out')
        im=nib.load(self.root/'ct_out/image.nii.gz')
        self.assertEqual(im.shape,(3,2,2))
        np.testing.assert_array_equal(np.asanyarray(im.dataobj)[:,:,0],-998)
        np.testing.assert_array_equal(np.asanyarray(im.dataobj)[:,:,1],-994)
        np.testing.assert_allclose(im.affine@np.array([1,1,1,1]),[-12,-23,32,1])
        self.assertEqual(meta['status'],'converted')

    def test_ct_missing_pixels_rejects_entire_series(self):
        a=self.ct('a.dcm',(0,0,0));b=self.ct('b.dcm',(0,0,1))
        import pydicom
        ds=pydicom.dcmread(b);ds.PixelData=b'';ds.save_as(b,enforce_file_format=True)
        with self.assertRaises(ConversionError):convert_series([a,b],self.root/'bad')
        self.assertFalse((self.root/'bad/image.nii.gz').exists())

    def test_ct_gap_rejected(self):
        pp=[self.ct(f'{i}.dcm',(0,0,z)) for i,z in enumerate([0,1,3])]
        with self.assertRaises(ConversionError):convert_series(pp,self.root/'gap')

    def test_ct_duplicate_position_rejected(self):
        pp=[self.ct(f'{i}.dcm',(0,0,0)) for i in range(2)]
        with self.assertRaises(ConversionError):convert_series(pp,self.root/'dup')

    def test_ct_changed_orientation_rejected(self):
        a=self.ct('a.dcm',(0,0,0));b=self.ct('b.dcm',(0,0,1),orientation=(0,1,0,1,0,0))
        with self.assertRaises(ConversionError):convert_series([a,b],self.root/'orient')

    def test_ct_tilt_preserves_affine(self):
        pp=[self.ct(f'{i}.dcm',(i*.2,0,i)) for i in range(3)]
        convert_series(pp,self.root/'tilt')
        im=nib.load(self.root/'tilt/image.nii.gz')
        np.testing.assert_allclose(im.affine[:3,2],[-.2,0,1],atol=1e-6)

    def test_us_uint16_static_png_is_lossless(self):
        pixels=np.array([[0,400,65535],[100,1000,9000]],dtype=np.uint16)
        p=self.root/'us.dcm';dicom(p,pixels,modality='US')
        result=convert_file(p,self.root/'usout')
        np.testing.assert_array_equal(np.array(Image.open(self.root/'usout/image.png')),pixels)
        self.assertEqual(result['kind'],'still')

    def test_us_video_keeps_all_frames_and_variable_timing(self):
        pixels=np.arange(18,dtype=np.uint8).reshape(3,2,3)
        p=self.root/'us.dcm';ds=dicom(p,pixels,modality='US',frames=3)
        ds.FrameTimeVector=[0,25,40];ds.FrameIncrementPointer=[0x00181065]
        ds.FrameDelay=12;ds.save_as(p,enforce_file_format=True)
        convert_file(p,self.root/'usout')
        with np.load(self.root/'usout/frames.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['frames'],pixels)
            np.testing.assert_allclose(z['timestamps_ms'],[0,25,65])
        side=json.loads((self.root/'usout/metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(side['timing']['frame_delay_ms'],12)

    def test_us_one_frame_multiframe_object_stays_video(self):
        p=self.root/'us.dcm';dicom(p,np.arange(6,dtype=np.uint8).reshape(2,3),modality='US',frames=1)
        r=convert_file(p,self.root/'usout')
        self.assertEqual(r['kind'],'video')
        with np.load(self.root/'usout/frames.npz') as z:self.assertEqual(z['frames'].shape,(1,2,3))

    def test_us_unknown_timing_not_invented(self):
        from pydicom.dataset import Dataset
        d=Dataset();d.RecommendedDisplayFrameRate=25
        t=frame_timing(d,4)
        self.assertIsNone(t['timestamps_ms'])
        self.assertEqual(t['recommended_display_fps'],25)

    def test_us_bad_vector_rejected(self):
        from pydicom.dataset import Dataset
        d=Dataset();d.FrameTimeVector=[0,20];d.FrameIncrementPointer=[0x00181065]
        with self.assertRaises(ConversionError):frame_timing(d,4)

    def test_existing_and_nested_output_rejected(self):
        src=self.root/'source';src.mkdir()
        with self.assertRaises(ValueError):prepare_output([src],src/'converted')
        out=self.root/'existing';out.mkdir()
        with self.assertRaises(FileExistsError):prepare_output([src],out)

    def test_ct_unreadable_header_blocks_conversion(self):
        from convert_ct import main
        src=self.root/'source';src.mkdir()
        for i in range(2):
            dicom(src/f'{i}.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,i),study=self.study,series=self.series)
        (src/'2.dcm').write_bytes(b'broken DICOM header')
        out=self.root/'run'
        self.assertEqual(main(['--input',str(src),'--output',str(out)]),2)
        self.assertFalse(list(out.rglob('image.nii.gz')))

    def test_ct_missing_series_uid_blocks_conversion(self):
        from convert_ct import main
        src=self.root/'source';src.mkdir()
        for i in range(3):
            ds=dicom(src/f'{i}.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,i),study=self.study,series=self.series)
            if i==2:
                del ds.SeriesInstanceUID;ds.save_as(src/f'{i}.dcm',enforce_file_format=True)
        out=self.root/'run'
        self.assertEqual(main(['--input',str(src),'--output',str(out)]),2)
        self.assertFalse(list(out.rglob('image.nii.gz')))

    def test_ct_missing_modality_blocks_conversion(self):
        from convert_ct import main
        src=self.root/'source';src.mkdir()
        for i in range(3):
            ds=dicom(src/f'{i}.dcm',np.ones((2,3),dtype=np.int16),position=(0,0,i),study=self.study,series=self.series)
            if i==2:
                del ds.Modality;ds.save_as(src/f'{i}.dcm',enforce_file_format=True)
        out=self.root/'run'
        self.assertEqual(main(['--input',str(src),'--output',str(out)]),2)
        self.assertFalse(list(out.rglob('image.nii.gz')))

    def test_ct_failed_write_does_not_publish_output(self):
        pp=[self.ct(f'{i}.dcm',(0,0,i)) for i in range(2)]
        with patch('convert_ct.make_preview',side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):convert_series(pp,self.root/'out')
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.incomplete-*')))

    def test_us_failed_write_does_not_publish_output(self):
        p=self.root/'us.dcm';dicom(p,np.ones((2,3),dtype=np.uint8),modality='US')
        with patch('convert_us.preview',side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):convert_file(p,self.root/'out')
        self.assertFalse((self.root/'out').exists())
        self.assertFalse(list(self.root.glob('.incomplete-*')))

    def test_ct_declared_spacing_mismatch_rejected(self):
        import pydicom
        pp=[self.ct(f'{i}.dcm',(0,0,i*2)) for i in range(3)]
        for p in pp:
            d=pydicom.dcmread(p);d.SpacingBetweenSlices=1;d.save_as(p,enforce_file_format=True)
        with self.assertRaises(ConversionError):convert_series(pp,self.root/'out')

    def test_us_rgb_video_keeps_channels(self):
        p=self.root/'rgb.dcm'
        d=dicom(p,np.zeros((2,2,3),dtype=np.uint8),modality='US',frames=2)
        a=np.arange(36,dtype=np.uint8).reshape(2,2,3,3)
        d.Rows=2;d.Columns=3;d.SamplesPerPixel=3;d.PlanarConfiguration=0
        d.PhotometricInterpretation='RGB';d.PixelData=a.tobytes();d.FrameTime=20
        d.save_as(p,enforce_file_format=True)
        convert_file(p,self.root/'out')
        with np.load(self.root/'out/frames.npz') as z:
            np.testing.assert_array_equal(z['frames'],a)
            np.testing.assert_array_equal(z['timestamps_ms'],[0,20])


if __name__=='__main__':unittest.main()
