from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class ReportingTests(unittest.TestCase):
    def test_report_uses_terminal_records_and_escapes_private_text(self):
        self.assertTrue((ROOT/'src/dataset1282_analysis/reporting.py').exists(),'report implementation missing')
        from dataset1282_analysis.reporting import aggregate_results,write_csv
        rows=[{'case_id':'A','status':'complete','metadata_link_status':'unmatched',
               'geometry':{'array_geometry_status':'match','physical_geometry_status':'unverified'},
               'image':{'mean':3,'nonfinite_count':0},'label':{'classes':[{'value':1,'voxels':5}], 'foreground_voxels':5}},
              {'case_id':'B','status':'failed','errors':[{'error':'<script>bad</script>'}]}]
        summary=aggregate_results(rows,expected_total=3)
        self.assertEqual(summary['denominators']['expected_pairs'],3)
        self.assertEqual(summary['denominators']['terminal_results'],2)
        self.assertEqual(summary['denominators']['pending'],1)
        self.assertEqual(summary['denominators']['image_statistics_available'],1)
        self.assertIsNone(summary['case_rows'][1]['image_mean'])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'table.csv'
            write_csv(p,[{'value':'=1+1','blank':None,'number':3}])
            text=p.read_text(encoding='utf-8-sig')
            self.assertIn("'=1+1",text)
            self.assertNotIn('None',text)


if __name__=='__main__':unittest.main()
