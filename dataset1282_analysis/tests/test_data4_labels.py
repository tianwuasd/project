import unittest

from dataset1282_analysis.prediction.data4 import label_semantics


class LabelSemanticsTests(unittest.TestCase):
    # CHD1282原训练及容器转换使用的同一套解剖编号。
    original={'background':0,'left_ventricle':1,'right_ventricle':2,'left_atrium':3,
              'right_atrium':4,'aorta':5,'pulmonary_artery':6,'myocardium':7,
              'superior_vena_cava':8,'inferior_vena_cava':9,'pulmonary_veins':10}
    converted={'background':0,'LV':1,'RV':2,'LA':3,'RA':4,'AO':5,'PA':6,'Myo':7,
               'SVC':8,'IVC':9,'PV':10}

    def test_original_and_converted_names(self):
        self.assertEqual(label_semantics(self.converted),self.original)
        self.assertEqual(label_semantics(self.original),self.original)

    def test_dictionary_order_does_not_change_semantics(self):
        self.assertEqual(label_semantics(dict(reversed(list(self.converted.items())))),self.original)

    def test_swapped_structure_ids_remain_different(self):
        changed=dict(self.converted,LV=2,RV=1)
        self.assertNotEqual(label_semantics(changed),label_semantics(self.original))

    def test_unknown_or_duplicate_semantics_are_rejected(self):
        for name in ('unverified_structure','left_ventricle'):
            changed=dict(self.converted)
            changed[name]=changed.pop('RV')
            with self.subTest(name=name),self.assertRaises(ValueError):
                label_semantics(changed)

    def test_invalid_structure_ids_are_rejected(self):
        for changed in (dict(self.converted,LV=True),dict(self.converted,LV=2),
                        dict(self.converted,background=1,LV=0)):
            with self.subTest(labels=changed),self.assertRaises(ValueError):
                label_semantics(changed)


if __name__=='__main__':
    unittest.main()
