from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import json
import os

coco = Coco()
# path = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/datasets_final/souvenir_dataset_1/souvenir_dataset/final_dataset/train/original_images"

# Opening JSON file
f = open("/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/label.json")
# returns JSON object as
data = json.load(f)

coco.add_category(CocoCategory(id=0,  name='shipping_box'))

# coco.add_category(CocoCategory(id=0,  name='cmu_tartan_bottle'))
# coco.add_category(CocoCategory(id=1,  name='all_start_dogs_belt_with_ball'))
# coco.add_category(CocoCategory(id=2,  name='cmu_cup'))
# coco.add_category(CocoCategory(id=3,  name='cmu_bottle'))
# coco.add_category(CocoCategory(id=4,  name='monkey_keychain'))
# coco.add_category(CocoCategory(id=5,  name='transparent_bottle'))
# coco.add_category(CocoCategory(id=6,  name='all_star_dogs_belt'))
# coco.add_category(CocoCategory(id=7,  name='table_tennis_balls'))
# coco.add_category(CocoCategory(id=8,  name='dog_keychain'))
# coco.add_category(CocoCategory(id=9,  name='binnie'))
# coco.add_category(CocoCategory(id=10, name='unicorn'))
# coco.add_category(CocoCategory(id=11, name='airpods_case'))

# coco.add_category(CocoCategory(id=0,  name='cmu_tartan_bottle'))
# coco.add_category(CocoCategory(id=1,  name='tennis_ball_toy'))
# coco.add_category(CocoCategory(id=2,  name='cmu_cup'))
# coco.add_category(CocoCategory(id=3,  name='cmu_bottle'))
# coco.add_category(CocoCategory(id=4,  name='monkey_keychain'))
# coco.add_category(CocoCategory(id=5,  name='transparent_bottle'))
# coco.add_category(CocoCategory(id=6,  name='all_star_dogs_belt'))
# coco.add_category(CocoCategory(id=7,  name='dog_collar'))
# coco.add_category(CocoCategory(id=8,  name='cow_keychain'))
# coco.add_category(CocoCategory(id=9,  name='beanie'))
# coco.add_category(CocoCategory(id=10, name='unicorn'))
# coco.add_category(CocoCategory(id=11, name='airpods_case'))


objects = ['shipping_box']

# objects = ['cmu_tartan_bottle','all_start_dogs_belt_with_ball','cmu_cup','cmu_bottle',\
#             'monkey_keychain','transparent_bottle','all_star_dogs_belt','table_tennis_balls',\
#             'dog_keychain','binnie','unicorn','airpods_case']

for key, value in data.items():
    coco_image = CocoImage(file_name=key, height=480, width=640)
    print("value : ",value)
    for name,annotation in value.items():
        width = annotation[1][0]-annotation[0][0]
        height = annotation[1][1]-annotation[0][1]
            
        coco_image.add_annotation(
            CocoAnnotation(
            bbox=[annotation[0][0],annotation[0][1],width, height],
            category_id=int(name),
            category_name=objects[int(name)]
            )
        )
    
    coco.add_image(coco_image)

save_path = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/staple_box/final_dataset/labels_coco.json"
save_json(data=coco.json, save_path=save_path)

