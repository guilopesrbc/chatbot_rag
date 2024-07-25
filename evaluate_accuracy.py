import getpass
import pandas as pd
from vectors_db import generate_data_store
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.evaluation import load_evaluator
import os


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

openai_api_key = getpass.getpass("OpenAI API Key: ")

questions = [
    "Quais são os critérios de avaliação de candidatos para o vestibular Unicamp 2025?",
    "Qual é o valor da taxa de inscrição para o vestibular Unicamp 2025?",
    "Quais são as políticas de cotas raciais e sociais no vestibular Unicamp 2025?",
    "Como funciona a correção das redações no vestibular da Unicamp?",
    "Quais são os documentos necessários para a inscrição no vestibular Unicamp 2025?",
    "Qual é o período de inscrição para o vestibular Unicamp 2025?",
    "Quantas vagas estão disponíveis para o curso de sistemas de informação?",
    "Como é a distribuição de vagas nos Cursos de Graduação da Unicamp?",
]

answers = [
    "Capacidade de se expressar com clareza; Capacidade de organizar suas ideias; Capacidade de estabelecer relações; Capacidade de interpretar dados e fatos; Capacidade de elaborar hipóteses; Domínio dos conteúdos das áreas de conhecimento desenvolvidas no Ensino Médio; Capacidade de relacionar e interpretar informações de caráter interdisciplinar, a partir das áreas de conhecimento presentes no Ensino Médio.",
    "R$ 210,00 (duzentos e dez reais)",
    "Os(as) candidatos(as) autodeclarados(as) pretos(as) e pardos(as) que optarem pela reserva de vagas (cotas) deverão preencher o campo específico de autodeclaração no formulário de inscrição,  A validação da autodeclaração, apresentada pelos(as) candidatos(as) optantes pelas cotas étnico-raciais, somente ocorrerá após a avaliação de fenótipo realizada pela Comissão de Averiguação",
    "A prova de Redação busca avaliar habilidades de leitura e escrita dos(as) candidatos(as) na produção de textos pertencentes a diferentes gêneros discursivos. Cada uma das Propostas de redação é acompanhada de tarefas a serem cumpridas pelos(as) candidatos(as) e de um ou mais textos para leitura, que visam subsidiar, respectivamente, a proposta temática e o seu projeto de texto. Ao propor gêneros discursivos, a prova de Redação procura simular situações reais de escrita, por isso é importante que os(as) candidatos(as) fiquem atentos à situação de produção e circulação do texto a ser elaborado e à interlocução dos gêneros discursivos solicitados na prova. Em geral, para que um texto seja bem-sucedido é preciso que os(as) candidatos(as) demonstrem ter experiência de leitura e saibam delinear um projeto de texto em função de um ou mais objetivos específicos, que deverão ser cumpridos por meio da elaboração escrita. A avaliação dos textos produzidos levará em conta: o cumprimento da proposta temática, a configuração do gênero (a sua situação de produção, circulação e interlocução), a qualidade da leitura dos textos oferecidos na prova, e a articulação coerente e coesa de elementos da escrita.",
    "Para todos(as) os(as) candidatos(as): Foto digital modelo 3x4; Diploma ou Certificado de Conclusão do ensino médio ou equivalente.",
    "01 a 30 de agosto de 2024",
    "50",
    "2537 vagas oferecidas pelo Vestibular Unicamp (VU) 2025; 314 vagas oferecidas pelo Edital ENEM-Unicamp 2025; 325 vagas oferecidas pelo Provão Paulista 2025; 49 vagas oferecidas pelo Vestibular Indígena (VI) 2025. O Vestibular Indígena terá ainda 81 vagas adicionais, conforme Edital a ser publicado, respeitando os princípios da Deliberação CONSU-A-032/2017; 115 vagas oferecidas pelo Edital de olimpíadas científicas e competições de conhecimento de áreas específicas. Haverá, ainda, 16 vagas adicionais nesse sistema de ingresso, conforme Edital a ser publicado, respeitando os princípios da Deliberação CONSU-A-032/2017."
]

evaluation_dt = pd.DataFrame({"Questions": questions, "Answers": answers})
chat_answers = {}

db = generate_data_store(api_key=openai_api_key)

for index, query in enumerate(questions):
    results = db.similarity_search_with_relevance_scores(query, k=3)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
    response = model.invoke(prompt)
    response_text = response.content
    
    chat_answers[index] = response_text

evaluation_dt["Chatbot Answers"] = [chat_answers[i] for i in range(len(questions))]

os.environ["OPENAI_API_KEY"] = openai_api_key

evaluator = load_evaluator(openai_api_key=openai_api_key, evaluator="pairwise_embedding_distance")

accuracy_score = {}
for i in range(len(evaluation_dt)):
    distance = evaluator.evaluate_string_pairs(prediction=evaluation_dt["Answers"][i], prediction_b=evaluation_dt["Chatbot Answers"][i])
    accuracy_score[i] = float(1 - float(distance['score']))

evaluation_dt["Accuracy"] = [accuracy_score[i] for i in range(len(questions))]
print(evaluation_dt)

evaluation_dt.to_csv("evaluation_dt.csv")
print(f"\nOverall Accuracy: {evaluation_dt["Accuracy"].mean()}")
