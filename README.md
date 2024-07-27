# RAG Chatbot Vestibular da Unicamp 2025

## Visão Geral do Projeto

Este projeto implementa um chatbot baseado em Retrieval Augmented Generation (RAG) utilizando Modelos de Linguagem de Grande Escala (LLM) para responder perguntas relacionadas ao vestibular da Unicamp 2025. O chatbot combina recuperação de documentos relevantes em relação à dúvida enviada pelo usuário com a geração de respostas baseadas nesses documentos, fazendo uso de processamento de Natural Language Processing (NLP) para geração de respostas mais precisas e contextualizadas.

## Implementação
O projeto foi desenvolvido a partir da linguagem de programação Python, se utilizando da publicação da Resolução GR-029/2024, de 10/07/2024 que "Dispõe sobre o Vestibular Unicamp 2025 para vagas no ensino de Graduação" como base de conhecimento, a fim de construir um Assistente Virtual (ChatBot) baseados nos conceitos de RAG para geração de prompts contextuais que serão enviados e respondidos via consumo da API do Modelo de Linguagem de Larga Escala ChatGPT da OpenAI.    
### Bibliotecas Utilizadas

- **python-dotenv**
- **langchain**
- **PyPDF2**
- **faiss-cpu**
- **pandas**

### Arquitetura do Projeto

1. **Base de conhecimento e indexação**: O documento "procuradoria_geral_normas.pdf" é alocado no diretório "data" e extraído no código com áuxilio da biblioteca PyPDF2. Posteriormente, o documento é fragmentado em trechos de até 1000 caracteres, chamados "chunks" e utilizado da biblioteca FAISS, que permite rápida recuperação via similaridade de dados vetoriais (Embeddings), para indexar os documentos de forma eficiente.
2. **Embeddings**: Os embeddings são gerados usando a API do OpenAI, transformam os chunks em conjuntos de dados vetoriais e são armazenados com a ajuda do FAISS.
3. **Pipeline de Recuperação**: O LangChain é utilizado para construir o pipeline de recuperação, fazendo a busca de similaridade de chunks no conjunto de dados baseado no embendding da pergunta enviada.
4. **Modelo de linguagem**: Com o retorno de chunks de maior similaridade o prompt contextualizado é construído e enviado via API para o modelo de linguagem OpenAI gpt-3.5-turbo. Retornando por fim uma resposta ao usuário.
5. **Interface e Deploy**: A interface de usuário e deploy foram desenvolvidas via Streamlit.

## Deploy do projeto
Disponível no [link](https://rag-chatbot-guilopesrbc.streamlit.app/), PS: necessário o fornecimento de OpenAI API Key.
![image](https://github.com/user-attachments/assets/e8761af4-f1c2-47ca-b083-e8f6edd28294)


## Avaliação de desempenho
A avaliação de desempenho do chatbot foi feita a partir do script, disponível [aqui](https://github.com/guilopesrbc/chatbot_rag/blob/main/evaluate_accuracy.py), que analisa a acurácia da resposta do chatbot comparada às respostas reais acerca da mesma pergunta, essa avaliação é feita via similaridade de embeddings entre as duas respostas e a acurácia é definida e normalizada utilizando o cálculo (1 - score). O resultado dessa avaliação é armazenado em um [dataframe](https://github.com/guilopesrbc/chatbot_rag/blob/main/evaluation_dt.csv) via pandas e calculado a média para acurácia total. A Acurácia média foi de aproximadamente 86%.

## Como testar localmente
Para realizar testes localmente, siga estas etapas abaixo:

Clone o Repositório:
```
git clone https://github.com/guilopesrbc/chatbot_rag.git
```
Instale as dependências:
```
pip install -r requirements.txt
```
### Testar ChatBot APP
Execute o streamlit:
```
streamlit run main.py
```
### Executar avaliação de desempenho
Execute o arquivo python "[evaluate_accuracy.py](https://github.com/guilopesrbc/chatbot_rag/blob/main/evaluate_accuracy.py)"

Em ambos os testes é necessário o fornecimento de OpenAI API Key, para teste no ChatBot APP bastar fornecer na caixa de texto solicitante, já na avaliação de desempenho deve ser colocada via terminal. 
