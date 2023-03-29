import pytest
from icevision.all import *

@pytest.fixture()
def setup_pred():
    # synthetic data to test
    gt = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    pred = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction


# testing metric 
@pytest.fixture()
def expected_binary_dice_output():

    return {'dummy_value_for_fastai': 0.4}


# @pytest.fixture()
def test_binary_dice_metric(setup_pred, expected_binary_dice_output):

    pred = setup_pred

    dice = BinaryDiceCoefficient()
    dice.accumulate([pred])

    assert dice.finalize() == expected_binary_dice_output