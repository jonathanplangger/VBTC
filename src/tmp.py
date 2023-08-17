import segmentation_models_pytorch as smp 
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'

model = smp.DeepLabV3Plus(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes = 19, 
    activation = "sigmoid"
)

# preprocessing function for the input 
preprocess_input = smp.encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

print(type(model))





