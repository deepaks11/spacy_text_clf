import spacy

nlp = spacy.load("test_model_0.0")
stop = True
while stop:
    test_text = input("Enter the Question : ")
    if test_text.strip() == "stop":
        stop = False
    else:
        doc = nlp(test_text)
        m = max(doc.cats.values())
        for k, v in doc.cats.items():
            if m == v:
                print("\n Found Intent : ", k, "\nScore : ", v)


