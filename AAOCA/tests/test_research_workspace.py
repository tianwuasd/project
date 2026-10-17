import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

module_path=Path(__file__).resolve().parents[1]/'tools/research_ct/workspace.py'
spec=importlib.util.spec_from_file_location('research_workspace',module_path)
workspace=importlib.util.module_from_spec(spec);spec.loader.exec_module(workspace)


class ResearchWorkspaceTests(unittest.TestCase):
    def test_detect_moved_root_and_keep_relative_outputs(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{},clear=True):
            root=Path(d)/'new_location';(root/'results/docs').mkdir(parents=True)
            (root/'CT总清单.json').write_text('[]')
            self.assertEqual(workspace.workspace_root(root/'results/docs/tool.py'),root.resolve())
            value={'output':'AAOCA/example/image.nii.gz'}
            self.assertEqual(workspace.relocate(value,root),value)

    def test_prefix_boundaries_and_most_specific_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            config={'path_mappings':[{'old':'D:/old','new':'D:/new'},
                                     {'old':'D:/old/nested','new':'D:/separate'}]}
            (root/'project.local.json').write_text(json.dumps(config))
            data={'a':r'd:\OLD\nested\image.nii.gz','b':'D:/older/image.nii.gz',
                  'c':['D:/old/file','patient_001']}
            result=workspace.relocate(data,root)
            self.assertEqual(result['a'],'D:/separate/image.nii.gz')
            self.assertEqual(result['b'],data['b'])
            self.assertEqual(result['c'],['D:/new/file','patient_001'])

    def test_explicit_root_takes_priority(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'CHD_PROJECT_ROOT':d}):
            self.assertEqual(workspace.workspace_root(__file__),Path(d).resolve())
