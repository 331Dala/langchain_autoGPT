import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = apikey

# App framework
st.title('🦜🔗GPT Creator')
prompt = st.text_input('Input your prompt here:')

# Prompt template
title_template = PromptTemplate(
	input_variables=["topic"],
	template="Write a title for the following topic: {topic}"
)

script_template = PromptTemplate(
	input_variables=["title", "wikipedia_research"],
	template="Write me a script based on this topic: {title} while leveraging this wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")


# llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_memory)

wiki = WikipediaAPIWrapper()

# sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)
# sequential_chain = SequentialChain(
# 	chains=[title_chain, script_chain], 
# 	input_variables=["topic"], 
# 	output_variables=['title', 'script'], 
# 	verbose=True) # get multiple outputs 

# Show stuff to the screen if there is a prompt
if prompt:
	# response = title_chain.run(prompt)
	# response = sequential_chain.run({'topic': prompt})
	title = title_chain.run(prompt)
	wikipedia_research = wiki.run(prompt)
	script = script_chain.run(title=title, wikipedia_research=wikipedia_research)
	# st.write(response)
	# st.write(response['title'])
	# st.write(response['script'])
	st.write(title)
	st.write(script)

	# with st.expander('Message History'):
		# st.info(memory.buffer)
	with st.expander('Title History'):
		st.info(title_memory.buffer)
	with st.expander('Script History'):
		st.info(script_memory.buffer)
	with st.expander('Wikipedia Research'):
		st.info(script_memory.buffer)
