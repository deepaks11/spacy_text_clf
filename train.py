import pandas as pd
import spacy
from spacy.util import minibatch, compounding

from spacy.pipeline.textcat import Config, single_label_cnn_config
from spacy.training import Example


def train_model():
    nlp = spacy.blank('en')
    # nlp = en_core_web_sm.load() # or nlp=spacy.load("en_core_web_sm")

    config = Config().from_str(single_label_cnn_config)
    if "textcat" not in nlp.pipe_names:
         textcat = nlp.add_pipe('textcat', config=config, last=True)

    df = pd.read_excel("modified.xlsx")
    df['tuples'] = df.apply(lambda row: (row['question'], row['intent']), axis=1)

    # Converting tuple to List
    train = df['tuples'].tolist()
    texts, labels = zip(*train)

    lab = []

    for lm in labels:
        if lm not in lab:
            lab.append(lm)

    [textcat.add_label(l) for l in lab]
    print(textcat.labels)

    cats = []
    d = {}
    [d.update({x: False}) for x in labels]
    for y in labels:
        if y in d.keys():
            d[y] = True
            cats.append(d.copy())
            d[y] = False


    TrainX = texts
    TrainY = cats
    train_data = list(zip(TrainX, [{'cats': cats} for cats in TrainY]))
    n_iter = 50
    # Disabling other components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        print("Training the model...")

        # Performing training
        for i in range(n_iter):
            print("Epoch : {} ".format(i))
            losses = {}
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                # nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                #            losses=losses)

                example = []
                # Update the model with iterating each text
                for i in range(len(texts)):
                    doc = nlp.make_doc(texts[i])
                    example.append(Example.from_dict(doc, annotations[i]))

                # Update the model
                # nlp.update(example)
                nlp.update(example,annotates=annotations, drop=0.2, losses=losses, sgd=optimizer)

        nlp.to_disk("test_model_0.0")
train_model()