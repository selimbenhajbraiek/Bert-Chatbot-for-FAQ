from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
import torch

model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

hotel_context = "The Grand Horizon Hotel is a 5-star luxury hotel located in downtown Budapest, Hungary. Established in 2005, it features 250 rooms, 30 suites, and panoramic views of the Danube River. The hotel offers premium amenities including a rooftop restaurant, a full-service spa, a fitness center, and conference halls for business events. Guests can check in from 2 PM and check out until 12 PM. The Grand Horizon is known for its award-winning customer service and eco-friendly practices, such as using solar energy and providing electric car charging stations. Reservations can be made online or by contacting the front desk at +36 1 234 5678."

def faq_bot(question):
    context = hotel_context
    input_ids = tokenizer. encode(question, context)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_idx+1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    output = model(torch. tensor([input_ids]), token_type_ids = torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = ' '.join(tokens[answer_start: answer_end+1])
    else:
        print("I don't know how to answer this question, can you ask another one?")
    corrected_answer = ''
    for word in answer.split():
        if word[0:2] == '##':
            corrected_answer += word[2:]
        else:
            corrected_answer += ' ' + word
    return corrected_answer

faq_bot("When was the hotel established?")

faq_bot("Does the coffee shop offer vegan snacks?")