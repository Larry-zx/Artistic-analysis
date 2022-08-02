from model.network import *


class multitask_classifer(BaseNetwork):
    def __init__(self, init=True, cnn_pretrained=True, model_type='resnet18'):
        super(multitask_classifer, self).__init__()
        self.cnn = FeatureExtraction(cnn_pretrained, model_type)
        self.artistFC = FC(model_type, output_dim=129)
        self.genreFC = FC(model_type, output_dim=11)
        self.styleFC = FC(model_type, output_dim=27)
        if init:
            self.init_FC()

    def init_FC(self):
        self.artistFC.init_weights()
        self.genreFC.init_weights()
        self.styleFC.init_weights()

    def forward(self, image):
        feature = self.cnn(image)
        pre_artist = self.artistFC(feature)
        pre_genre = self.genreFC(feature)
        pre_style = self.styleFC(feature)

        return {'artist':pre_artist, 'genre':pre_genre, 'style':pre_style}


class singletask_classifer(BaseNetwork):
    def __init__(self, init=True, cnn_pretrained=True, model_type='resnet50'):
        super(singletask_classifer, self).__init__()
        self.artistCNN = FeatureExtraction(cnn_pretrained, model_type)
        self.genreCNN = FeatureExtraction(cnn_pretrained, model_type)
        self.styleCNN = FeatureExtraction(cnn_pretrained, model_type)
        self.artistFC = FC(model_type, output_dim=129)
        self.genreFC = FC(model_type, output_dim=11)
        self.styleFC = FC(model_type, output_dim=27)

        if init:
            self.init_FC()

    def init_FC(self):
        self.artistFC.init_weights()
        self.genreFC.init_weights()
        self.styleFC.init_weights()

    def forward(self, image):
        artist_feature = self.artistCNN(image)
        genre_feature = self.genreCNN(image)
        style_feature = self.styleCNN(image)

        pre_artist = self.artistFC(artist_feature)
        pre_genre = self.genreFC(genre_feature)
        pre_style = self.styleFC(style_feature)

        return {'artist':pre_artist, 'genre':pre_genre, 'style':pre_style}
