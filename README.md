Paper correction:

ForeNet applies the Transformer encoder. The paper describes that its computational complexity is O(1), which refers to the operation of a single self-attention mechanism. Due to the parallelization of deep learning, each variable runs a self-attention mechanism. Therefore, the total computational complexity is O(N).


Start :

install pip env by requirements.txt

you can obtain the Weather, Traffic, and Electricity benchmarks from Google Drive 'https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy' provided in paper Autoformer; obtain the Solar benchmark from 'https://drive.google.com/drive/folders/12ffxwxVAGM_MQiYpIk9aBLQrb2xQupT-'; then put them into 'dataset'; Since PEMS04 and PEMS07 exceed the upload size limit, we can only temporarily provide PEMS03 and PEMS08. Please look for 04 and 07 in relevant papers

training and test by running 'scripts\VariableTST\xxx.sh', which shows the complete hyper-parameters

note that ForeNet is renamed from VariableTST, they are the same
