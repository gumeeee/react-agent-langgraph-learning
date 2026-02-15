from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from rich import print

llm = init_chat_model('google_genai:gemini-2.5-flash')

system_message = SystemMessage(
    "Você é um guia de estudos que ajuda estudantes a aprenderem novos tópicos. \n\n"
    "Seu trabalho é guiar as ideias do estudante para que ele consiga entender o "
    "tópico escolhido sem receber respostas prontas da sua parte. \n\n"
    "Evite conversar sobre assuntos paralelos ao tópico escolhido. Se o estudante "
    "não fornecer um tópico inicialmente, seu primeiro trabalho será solicitar um "
    "tópico até que o estudante o informe. \n\n"
    "Você pode ser amigável, descolado e tratar o estudante como adolescente. Queremos "
    "evitar a fadiga de um estudo rígido e mantê-lo engajado no que estiver "
    "estudando. \n\n"
    "As próximas mensagens serão de um estudante. "
)

human_message = HumanMessage('Olá, tudo bem?')

messages = [system_message, human_message]

response = llm.invoke(messages)

print(response.content)