import unittest
from src.selective_evaluation import selective_summary

class SelectiveEvaluationTests(unittest.TestCase):
    def test_abstention_does_not_count_as_correct(self):
        result=selective_summary([0,1,2,2],[0,0,2,1],[True,False,True,False])
        self.assertEqual(result['coverage'],.5)
        self.assertEqual(result['accepted_accuracy'],1)
        self.assertEqual(result['errors_referred'],2)
        self.assertEqual(result['per_class_coverage']['1'],0)
    def test_empty_acceptance_has_undefined_risk(self):
        result=selective_summary([0,1,2],[1,1,0],[False]*3)
        self.assertIsNone(result['accepted_accuracy'])
        self.assertIsNone(result['selective_risk'])
    def test_accepting_all_preserves_errors(self):
        result=selective_summary([0,1,2],[0,0,2],[True]*3)
        self.assertAlmostEqual(result['selective_risk'],1/3)
        self.assertEqual(result['errors_accepted'],1)
    def test_invalid_inputs_fail(self):
        for args in [([],[],[]),([0],[0,1],[True]),([3],[0],[True])]:
            with self.assertRaises(ValueError): selective_summary(*args)
