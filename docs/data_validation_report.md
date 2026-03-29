#### Confirm images are organised into correct and incorrect subfolders

Data has been organised in this way

data/raw/
  train/
    correct/    ← class_id 0 images
    incorrect/  ← class_id 1 images
  test/
    correct/
    incorrect/
  valid/
    correct/
    incorrect/


#### Count total images per class and confirm there are no empty folders
── Organised dataset summary ──────────────────────
  train / correct     →  524 images
  train / incorrect   →  459 images
   test / correct     →  183 images
   test / incorrect   →  271 images
  valid / correct     →  173 images
  valid / incorrect   →  349 images
  TOTAL              →  1959 images

16 file(s) skipped (see warnings above)


#### Verify all files are valid image formats (jpg, jpeg, png) and not corrupt
Verified 

#### Check for approximate class balance — flag if one class has more than 2x the images of the other
Checked


#### Spot-check a sample of images from each class to visually confirm labels are accurate
Checked

#### Flag any images with missing labels or corrupt files
Flagged and skipped 

[WARN] Empty label file for 5dc9b425fd85fd145de9153c_girl-sitting-in-chair_jpg.rf.90252e549057da4b8fe5a5746b06717d.jpg — skipped
[WARN] Empty label file for back-b-1-1-400x600xc_jpg.rf.c0026a03cda1013613d6268f2b890917.jpg — skipped
[WARN] Empty label file for frame_0272_jpg.rf.73cdcf468da196fb48ff0d36fcc586c3.jpg — skipped
[WARN] Empty label file for frame_0281_jpg.rf.38c2ccf2ea0d43e8b18c31b2bf95639a.jpg — skipped
[WARN] Empty label file for frame_0497_jpg.rf.9ed212d7340864b3367e9d43d40f1267.jpg — skipped
[WARN] Empty label file for frame_0774_jpg.rf.261fc2d1e076c69a40da408e3cc2da6b.jpg — skipped
[WARN] Empty label file for frame_20031_jpg.rf.b4791130f32b952a03d0f676d129a3f4.jpg — skipped
[WARN] Empty label file for frame_20033_jpg.rf.a5cbfa03ac16e66b41924f53bb89371d.jpg — skipped
[WARN] Empty label file for frame_20039_jpg.rf.7348f680229f46f64b423c8ed88c803a.jpg — skipped
[WARN] Empty label file for frame_20219_jpg.rf.cee52bcf03d458ac01b285885a53b67d.jpg — skipped
[WARN] Empty label file for IMG_9854_jpg.rf.3f3dcf3558a2b398e0c28910a44c7f9a.jpg — skipped
[WARN] Empty label file for IMG_9856_jpg.rf.dc644c3d41f7ac396273d3cd1afc0439.jpg — skipped
[WARN] Empty label file for images_jpg.rf.1820e695e74cffd2d3de55a46c8ed725.jpg — skipped
[WARN] Empty label file for frame_0273_jpg.rf.e2d6d083e22745027118e4896c53e476.jpg — skipped
[WARN] Empty label file for frame_0274_jpg.rf.0d6f1710b54f215391a67e151f7795bb.jpg — skipped
[WARN] Empty label file for train05_png_jpg.rf.1dee23d8ac3b5fc1390c7fc878ce9721.jpg — skipped