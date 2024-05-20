import spacy
from spacy.tokens import Doc
from spacy.training import Example

nlp = spacy.blank("en")

# Add the NER pipe to the pipeline
ner = nlp.add_pipe("ner")

# Add the labels you want to train
ner.add_label("OBJECT")  # Change this to your desired label

# Prepare training data
sentences = [
    "Please replace the green button and text form placement.",
    "Please swap the positions of the input form and the red button.",
    "Please rearrange the blue button and red button.",
    "Please swap the text form and gray button placement.",
    "Please replace and place the light blue button and form.",
    "Please change the position of the input form and yellow button.",
    "Please reverse the order of the white button and yellow button.",

    "Please rearrange the positions of the text form and black button.",
    "Please change the position of the green button and gray button and place them on opposite sides.",
    "Please swap the positions of the light blue button and the black button.",
    "Add new button.",
    "Delete new button 2.",
    "Insert new button 3.", 
    "Move new button 4.",
    "Please add new button.",
    "Please delete new button2.",
    "Please insert new button3.", 
    "Please move new button4.",
]

# Define the annotations
annotations = [
    {"entities": [(19, 31, "OBJECT"), (36, 45, "OBJECT")]},
    {"entities": [(33, 43, "OBJECT"), (52,62, "OBJECT")]},
    {"entities": [(21, 32, "OBJECT"), (37, 47, "OBJECT")]},
    {"entities": [(16, 25, "OBJECT"), (30, 41, "OBJECT")]},
    {"entities": [(29, 46, "OBJECT"), (51, 55, "OBJECT")]},
    {"entities": [(34, 44, "OBJECT"), (49, 62, "OBJECT")]},
    {"entities": [(32, 44, "OBJECT"), (49, 62, "OBJECT")]},
    {"entities": [(38, 47, "OBJECT"), (52, 64, "OBJECT")]},
    {"entities": [(34, 46, "OBJECT"), (51, 62, "OBJECT")]},
    {"entities": [(33, 50, "OBJECT"), (59, 71, "OBJECT")]},
    {"entities": [(4, 14, "OBJECT")]},
    {"entities": [(7, 19, "OBJECT")]},
    {"entities": [(7, 19, "OBJECT")]},
    {"entities": [(5, 17, "OBJECT")]},
    {"entities": [(11, 21, "OBJECT")]},
    {"entities": [(14, 25, "OBJECT")]},
    {"entities": [(14, 25, "OBJECT")]},
    {"entities": [(12, 23, "OBJECT")]},
]

# Create a list of Example objects
examples = []
for sentence, annotation in zip(sentences, annotations):
    doc = nlp.make_doc(sentence)
    example = Example.from_dict(doc, annotation)
    examples.append(example)

# Train the model
optimizer = nlp.initialize()
for i in range(100):
    for example in examples:
        nlp.update([example], sgd=optimizer)

# Save the trained model
nlp.to_disk("OBJECT_model")

# Load the trained model
nlp = spacy.load("OBJECT_model")

# Define the text you want to evaluate
texts = [
    "Please replace the green button and text form placement.",
    "Please swap the positions of the input form and the red button.",
    "Please rearrange the blue button and red button.",
    "Please swap the text form and gray button placement.",
    "Please replace and place the light blue button and form.",
    "Please change the position of the input form and yellow button.",
    "Please reverse the order of the white button and yellow button.",
    "Please rearrange the positions of the text form and black button.",
    "Please change the position of the green button and gray button and place them on opposite sides.",
    "Please swap the positions of the light blue button and the black button.",
    "Delete new button.",
    "Delete new button 2.",
    "Insert new button 3.", 
    "Move new button 4.",
    "Please add new button.",
    "Please delete new button2.",
]

# Create a doc object for each text and print the non-labeled entities

for text in texts:
    doc = nlp(text)
    print(f"Text: {text}")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")
    print("\n")