from sklearn.utils import shuffle
def generator(samples, batch_size=32):

    num_samples = len(samples)

    while 1:
        shuffle(samples)

        for div_point in range (0, num_samples, batch_size):
            batch_samples = samples[div_point:div_point+batch_size]

            images = []
            labels = []

            for batch_sample in batch_samples:
                # here the extraction of the images and labels needs to go

                images.append() #insert used variable for single images
                labels.append() #insert used variable for single labels

            X_train = np.array(images)
            Y_train = np.array(labels)

            yield shuffle(X_train, Y_train)