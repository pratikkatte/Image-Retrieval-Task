# Near Duplicate Detection and Localization Task.

## Steps To Execute.
1.To install the dependecies in a virtual enviornment.
  ```
  pip install -r requirements.txt
 ```
2. To fetch data from the source. The following execution will download the images and query dataset and save in the data folder.
```
 python data.py
```
3. Fit Principal component analysis (PCA) model to the images dataset, in order to apply PCA whitening to the add input image. The goal of this step is to make the input image less redundant and all the pixels have same variance. The model is saved in features folder
```
  python learnpca.py
 ```
4. Feature representation of the input image. The output of the step is saved in features folder.
```
  python extractor.py
 ```

5. outputs a outputs.json file containing the retrieved images which are similar to the query dataset with location co-ordinates.
```
  python result.py
 ```

## Evaluate Model on Test Dataset.
1. To fetch data from the source. The following execution will download the images and query dataset and save in the data folder.
```
 python eval_data.py
```
2. Learn the PCA model on the dataset.
```
python learnpca.py --features-dir eval_features --image-dir eval_data/sample_testset/images/
```
3. Extract features of the images in the dataset.
```
python extractor.py --features-dir eval_features/ --image-dir eval_data/sample_testset/images
```
4. retrieve similar images and save in predictions.json
```
python result.py --features eval_features --image-dir eval_data/sample_testset/crops/ --output-file predictions.json
```
5. Evaluate Model
```
python model_evaluation.py 
```
