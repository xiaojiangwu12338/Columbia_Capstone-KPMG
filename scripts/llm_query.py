from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.response_generator import ResponseGenerator

llm_client = LLMClient(api_key="", 
                        model="gpt-5", 
                        provider="openai",
                        base_url = "https://api.bltcy.ai/v1")

response_generator = ResponseGenerator(llm_client)
qustion_1 = "When did redetrmination begin for the COVID-19 Public Health Emergency unwind in New York State"
response_1 = response_generator.answer_question(qustion_1)
print("Question 1: ", qustion_1,"\n")
print(response_1["answer"])
print("-"*100)

question_2 = "When did the public health emergency end?"
response_2 = response_generator.answer_question(question_2)
print("Question 2: ", question_2,"\n")
print(response_2["answer"])
print("-"*100)

question_3 = "When submitting a claim for Brixandi, how many units should be indicated on the claim?"
response_3 = response_generator.answer_question(question_3)
print("Question 3: ", question_3,"\n")
print(response_3["answer"])
print("-"*100)

question_4 = "What rate codes should FQHCs use to bill for audio only telehealth?"
response_4 = response_generator.answer_question(question_4)
print("Question 4: ", question_4,"\n")
print(response_4["answer"])
print("-"*100)

question_5 = "Give me a chronological list of the commissioners and what year they first appeared in the medicaid updates."
response_5 = response_generator.answer_question(question_5)
print("Question 5: ", question_5,"\n")
print(response_5["answer"])
print("-"*100)

question_6 = "What are the requirements for appointment scheduling in the medicaid model contract for urgent care?"
response_6 = response_generator.answer_question(question_6)
print("Question 6: ", question_6,"\n")
print(response_6["answer"])
print("-"*100)

question_7 = "When did the pharmacy carve out occur?"
response_7 = response_generator.answer_question(question_7)
print("Question 7: ", question_7,"\n")
print(response_7["answer"])
print("-"*100)

question_8 = "What are the key components of the SCN program in the NYHER Waiver?"
response_8 = response_generator.answer_question(question_8)
print("Question 8: ", question_8,"\n")
print(response_8["answer"])
print("-"*100)

question_9 = "What constitutes RRP referral requirements?"
response_9 = response_generator.answer_question(question_9)
print("Question 9: ", question_9,"\n")
print(response_9["answer"])
print("-"*100)

question_10 = "what are the requirements for a referral for enrollment in the childrens waiver?"
response_10 = response_generator.answer_question(question_10)
print("Question 10: ", question_10,"\n")
print(response_10["answer"])
print("-"*100)

question_11 = "What are REC services offered to NYS providers?"
response_11 = response_generator.answer_question(question_11)
print("Question 11: ", question_11,"\n")
print(response_11["answer"])
print("-"*100)
