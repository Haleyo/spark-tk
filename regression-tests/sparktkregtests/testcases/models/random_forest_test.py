""" Tests the random forest functionality """

import unittest
from sparktkregtests.lib import sparktk_test


class RandomForest(sparktk_test.SparkTKTestCase):

    def setUp(self):
        """Build the required frame"""
        super(RandomForest, self).setUp()

        schema = [("feat1", int), ("feat2", int), ("class", str)]
        filename = self.get_file("rand_forest_class.csv")

        self.frame = self.context.frame.import_csv(filename, schema=schema)

    def test_rand_forest_class(self):
        """Test binomial classification of random forest model"""
        rfmodel = self.context.models.classification.random_forest_classifier.train(
            self.frame, "class", ["feat1", "feat2"], seed=0)

        rfmodel.predict(self.frame)
        preddf = self.frame.download()
        for index, row in preddf.iterrows():
            self.assertEqual(float(row['class']), float(row['predicted_class']))

        test_res = rfmodel.test(self.frame, ["feat1", "feat2"])

        self.assertEqual(test_res.precision, 1.0)
        self.assertEqual(test_res.recall, 1.0)
        self.assertEqual(test_res.accuracy, 1.0)
        self.assertEqual(test_res.f_measure, 1.0)

        self.assertEqual(
            test_res.confusion_matrix['Predicted_Pos']['Actual_Pos'], 413)
        self.assertEqual(
            test_res.confusion_matrix['Predicted_Pos']['Actual_Neg'], 0)
        self.assertEqual(
            test_res.confusion_matrix['Predicted_Neg']['Actual_Pos'], 0)
        self.assertEqual(
            test_res.confusion_matrix['Predicted_Neg']['Actual_Neg'], 587)

    def test_rand_forest_regression(self):
        """Test binomial classification of random forest model"""
        rfmodel = self.context.models.classification.random_forest_classifier.train(
            self.frame, "class", ["feat1", "feat2"], seed=0)
        
        rfmodel.predict(self.frame)
        preddf = self.frame.download(self.frame.count())
        for index, row in preddf.iterrows():
            self.assertAlmostEqual(float(row['class']), row['predicted_class'])

    @unittest.skip("publish model does not yet exist")
    def test_rand_forest_publish(self):
        """Test binomial classification of random forest model"""
        self.context.models.classification.random_forest_classif.train(self.frame, "class", ["feat1", "feat2"], seed=0)
        path = rfmodel.publish()
        self.assertIn("hdfs", path)
        self.assertIn("tar", path)

    @unittest.skip("publish model does not yet exist")
    def test_rand_forest_publish_classifier(self):
        """Test binomial classification of random forest model"""
        rfmodel = ia.RandomForestClassifierModel()
        rfmodel.train(self.frame, "class", ["feat1", "feat2"], seed=0)
        path = rfmodel.publish()
        self.assertIn("hdfs", path)
        self.assertIn("tar", path)


if __name__ == '__main__':
    unittest.main()
