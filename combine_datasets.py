import os
import json
import cv2

datasets_dir = "/media/jigar/A4F2A156F2A12D8C/CMU/SEM_3/project/combine_datasets"

image_counter = 0
final_dataset_dir = os.path.join(datasets_dir,"final_dataset")
final_dataset_dir_original_images = os.path.join(final_dataset_dir,"train","original_images")
final_dataset_dir_labelled_images = os.path.join(final_dataset_dir,"train","labelled_images")

if not os.path.exists(final_dataset_dir):
	os.makedirs(final_dataset_dir)
if not os.path.exists(final_dataset_dir_original_images):
	os.makedirs(final_dataset_dir_original_images)
if not os.path.exists(final_dataset_dir_labelled_images):
	os.makedirs(final_dataset_dir_labelled_images)

final_labels_file = os.path.join(final_dataset_dir, "label.json")
final_labels = {}
for dataset_path in os.listdir(datasets_dir):
    if "final" not in dataset_path:
        dataset = os.path.join(datasets_dir,dataset_path)
        labelled_images = os.path.join(dataset,"drop_box_dataset","train","labelled_images")
        original_images = os.path.join(dataset,"drop_box_dataset","train","original_images")
        labels_file = os.path.join(dataset,"drop_box_dataset","train","labels.json")
        print("labels_file : ",labels_file)
        print("labelled_images : ",labelled_images)
        print("original_images : ",original_images)
        labels = json.load(open(labels_file))
        
        for image_name in os.listdir(labelled_images):
            labelled_image = os.path.join(labelled_images,image_name)
            original_image = os.path.join(original_images,image_name)
            print("labelled_image : ",labelled_image)
            print("original_image : ",original_image)
            label = labels[image_name]
            img_name = "raw_image_"+str(image_counter)+".jpg"
            final_labels[img_name] = label
            final_original_image = cv2.imread(original_image)
            final_labelled_image = cv2.imread(labelled_image)
            # import pdb; pdb.set_trace()
            cv2.imwrite(os.path.join(final_dataset_dir_labelled_images,img_name),final_labelled_image)
            cv2.imwrite(os.path.join(final_dataset_dir_original_images,img_name),final_original_image)
            image_counter += 1
            
outfile = os.path.join(final_labels_file)
with open(outfile, "w") as o:
	json.dump(final_labels, o,indent = 4)
            

