# steps to start pytorch serve

1 create a new conda env and  install the require packages

pip3 install -r requirements.txt

2  run the notebook in ./notebook/OCR_from_Images_with_Transformers.ipynb to the save the pretrained model to disk.
You should have json file and bin file inside the raw_model folder

3 Create .mar file using the shell script and  trocr-hand-written.mar will be created (Please note: this step will take sometime to complete)

sh ./mar_file_creator.sh 

4 Once the mar file is create move it to model_store directory

mkdir model_store

mv trocr-hand-written.mar  ./model_store/

5 Start serving

torchserve --start --no-config-snapshots --model-store model_store --models trocr-hand-written.mar

6 test using image of your choice using curl

curl -X POST http://127.0.0.1:8080/predictions/trocr-hand-written  -T a01-122-02.jpg

7 integrate this inside a fastapi

