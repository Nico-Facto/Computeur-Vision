import numpy as np 


class NeuroneReseau :

    def __init__(self,layer_sizes):
        poid_shape = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])]
        self.poids = [np.random.standard_normal(s)/s[1]**.5 for s in poid_shape]
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]

    def predict (self, a) :
        for w,b in zip(self.poids,self.biases):
            a = self.activation(np.matmul(w,a)+b)
        return a    

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        correct = sum([np.argmax(a) == np.argmax(b) for a,b in zip(predictions,labels)])
        print("{0}/{1} accuracy: {2}%".format(correct,len(images), (correct/len(images))*100))


    @staticmethod
    def activation(x):
        return 1/(1+np.exp(-x))                