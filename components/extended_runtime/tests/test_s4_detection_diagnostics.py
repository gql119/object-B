import unittest
from scripts.s4_detection_diagnostics import person_dog_counts


class DetectionDefinitions(unittest.TestCase):
    def test_duplicate_dogs_count_one_real_GT(self):
        d=person_dog_counts([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]],[11,11])
        self.assertEqual(d['Person_GT_with_Dog_without_Person_overlap'],1)

    def test_correct_person_excludes_dog_without_person_counter(self):
        d=person_dog_counts([[0,0,10,10]],[[0,0,10,10],[0,0,10,10]],[11,14])
        self.assertEqual(d['Person_GT_with_Dog_overlap'],1)
        self.assertEqual(d['Person_GT_with_Dog_without_Person_overlap'],0)

    def test_small_contained_box_is_not_high_IoU_and_noGT_is_zero(self):
        d=person_dog_counts([[0,0,10,10]],[[0,0,2,2]],[11])
        self.assertEqual(d['Person_GT_with_Dog_overlap'],0)
        self.assertEqual(person_dog_counts([],[],[])['Person_GT'],0)


if __name__=='__main__':unittest.main()
