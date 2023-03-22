#import requests
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import time
import requests
import json
from contextlib import closing
import urllib.request
from urllib.request import urlopen, URLError
import discord
from discord import app_commands
from discord.ext import commands, tasks
from bs4 import BeautifulSoup
import sys
import os

# Bot Token
TOKEN = 'Bot Key'

# Ai
os.environ["OPENAI_API_KEY"] = 'Ai Key'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

# Making the discord bot
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)
bot.remove_command("help")

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"{len(synced)} commands synced!")
    except:
        print("A sync error occured")
    # UND Discord
    #test_ch = bot.get_channel(1044129340432056361)
    # Harvard Discord
    test_ch = bot.get_channel(915058337333276705)
    await test_ch.send(f"***{bot.user} has Started***")
    await test_ch.send(f"{len(synced)} commands synced")

@bot.tree.command(name="prompt", description="Ask a question!")
async def metar_cmd(interaction: discord.interactions, question: str):
    await interaction.response.defer()
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(question, response_mode="compact")
    await interaction.followup.send(f"```{response.response}```")

@bot.tree.command(name="update", description="Updates Ai Index")
@commands.has_permissions(administrator=True)
async def metar_cmd(interaction: discord.interactions):
    await interaction.response.defer()
    index = construct_index("docs")
    await interaction.followup.send("The index has been updated")


bot.run(TOKEN)