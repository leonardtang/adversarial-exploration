import matplotlib.pyplot as plt
from pycocotools.coco import COCO

dataDir = '/Users/leonard/Desktop/coco'
dataType = 'val2014'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)


def get_masked_images():
    coco = COCO(annFile)
    for i in range(1, 81):
        category_IDs = coco.getCatIds(catNms=i)
        imgIds = coco.getImgIds(catIds=category_Ids)
