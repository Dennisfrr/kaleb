import asyncio
import time
import os
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

# --- Configuração ---
# Token do Discord vindo do ambiente (mais seguro)
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")

# Endereço da API (permite override por ambiente)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --- Lógica do Bot ---

# Para que o bot possa receber mensagens, precisamos definir as 'Intents'.
# Intents são como permissões que dizem ao Discord quais eventos seu bot quer ouvir.
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
http_session: aiohttp.ClientSession | None = None
user_cooldowns: dict[int, float] = {}
COOLDOWN_SECONDS = 3.0

@bot.event
async def on_ready():
    """Chamado quando o bot se conecta com sucesso ao Discord."""
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))

    try:
        await bot.tree.sync()
        print("Slash commands sincronizados.")
    except Exception as e:
        print(f"Falha ao sincronizar slash commands: {e}")

    print(f'Bot conectado como {bot.user}')
    print('------')
    print('Comandos de texto:')
    print('!ask <sua pergunta>  |  !learn <tópico>')
    print('Slash commands: /ask, /learn, /ping')
    print('------')

@bot.event
async def on_message(message: discord.Message):
    """Esta função é chamada toda vez que uma mensagem é enviada em um canal que o bot pode ver."""
    # Ignora mensagens enviadas pelo próprio bot para evitar loops infinitos.
    if message.author == bot.user:
        return

    # --- Comando !ask ---
    if message.content.startswith('!ask '):
        question = message.content[5:].strip() # Remove '!ask ' do início da mensagem
        if not question:
            await message.channel.send("Por favor, digite uma pergunta após o comando `!ask`.")
            return

        # Cooldown simples por usuário
        now = time.time()
        last = user_cooldowns.get(message.author.id, 0)
        if now - last < COOLDOWN_SECONDS:
            await message.channel.send("⏳ Aguarde um pouco antes de usar o comando novamente.")
            return
        user_cooldowns[message.author.id] = now

        # "digitando" para feedback
        await message.channel.trigger_typing()

        try:
            assert http_session is not None
            session_id = f"guild:{message.guild.id if message.guild else 'dm'}:user:{message.author.id}"
            payload = {"message": question, "session_id": session_id}
            async with http_session.post(f"{API_BASE_URL}/api/chat", json=payload) as resp:
                if resp.status == 200:
                    api_data = await resp.json()
                    answer = api_data.get("reply", "Não recebi uma resposta válida da API.")
                    embed = discord.Embed(title="Resposta", description=answer, color=0x2ecc71)
                    embed.set_footer(text="Sistema Cortical")
                    await message.channel.send(embed=embed)
                else:
                    try:
                        api_err = await resp.json()
                        detail = api_err.get('detail', 'sem detalhes')
                    except Exception:
                        detail = 'sem detalhes'
                    await message.channel.send(f"😥 Erro ao contatar a API (Código: {resp.status}).\nDetalhe: {detail}")

        except aiohttp.ClientError as e:
            await message.channel.send(f"😥 Não consegui me conectar à API. Verifique se `api.py` está rodando.\nErro: {e}")

    # --- Comando !learn ---
    if message.content.startswith('!learn '):
        topic = message.content[7:].strip() # Remove '!learn ' do início da mensagem
        if not topic:
            await message.channel.send("Por favor, digite um tópico após o comando `!learn`.")
            return

        # Cooldown simples por usuário
        now = time.time()
        last = user_cooldowns.get(message.author.id, 0)
        if now - last < COOLDOWN_SECONDS:
            await message.channel.send("⏳ Aguarde um pouco antes de usar o comando novamente.")
            return
        user_cooldowns[message.author.id] = now

        await message.channel.trigger_typing()

        try:
            assert http_session is not None
            async with http_session.post(f"{API_BASE_URL}/learn", json={"topic": topic}) as resp:
                if resp.status == 200:
                    api_data = await resp.json()
                    confirmation_message = api_data.get("message", "O processo de aprendizado foi concluído.")
                    embed = discord.Embed(title="Aprendizado concluído", description=confirmation_message, color=0x3498db)
                    await message.channel.send(embed=embed)
                else:
                    try:
                        api_err = await resp.json()
                        detail = api_err.get('detail', 'sem detalhes')
                    except Exception:
                        detail = 'sem detalhes'
                    await message.channel.send(f"😥 Erro durante o aprendizado (Código: {resp.status}).\nDetalhe: {detail}")

        except aiohttp.ClientError as e:
            await message.channel.send(f"😥 Não consegui me conectar à API para aprender. Verifique se `api.py` está rodando.\nErro: {e}")

    # Permite que comandos de texto coexistam com slash commands
    await bot.process_commands(message)


# --- Slash Commands ---
@bot.tree.command(name="ask", description="Fazer uma pergunta ao sistema cortical")
async def slash_ask(interaction: discord.Interaction, pergunta: str):
    now = time.time()
    last = user_cooldowns.get(interaction.user.id, 0)
    if now - last < COOLDOWN_SECONDS:
        await interaction.response.send_message("⏳ Aguarde um pouco antes de usar o comando novamente.", ephemeral=True)
        return
    user_cooldowns[interaction.user.id] = now

    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        session_id = f"guild:{interaction.guild.id if interaction.guild else 'dm'}:user:{interaction.user.id}"
        payload = {"message": pergunta, "session_id": session_id}
        async with http_session.post(f"{API_BASE_URL}/api/chat", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                answer = data.get("reply", "Sem resposta da API.")
                embed = discord.Embed(title="Resposta", description=answer, color=0x2ecc71)
                embed.set_footer(text="Sistema Cortical")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


@bot.tree.command(name="learn", description="Ensinar um novo conceito ao sistema")
async def slash_learn(interaction: discord.Interaction, topico: str):
    now = time.time()
    last = user_cooldowns.get(interaction.user.id, 0)
    if now - last < COOLDOWN_SECONDS:
        await interaction.response.send_message("⏳ Aguarde um pouco antes de usar o comando novamente.", ephemeral=True)
        return
    user_cooldowns[interaction.user.id] = now

    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        async with http_session.post(f"{API_BASE_URL}/learn", json={"topic": topico}) as resp:
            if resp.status == 200:
                data = await resp.json()
                msg = data.get("message", "Aprendizado concluído.")
                embed = discord.Embed(title="Aprendizado", description=msg, color=0x3498db)
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


@bot.tree.command(name="ping", description="Verificar latência e saúde da API")
async def slash_ping(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True, ephemeral=True)
    latency_ms = round(bot.latency * 1000)
    api_ok = False
    try:
        assert http_session is not None
        async with http_session.get(f"{API_BASE_URL}/ping") as resp:
            api_ok = resp.status == 200
    except Exception:
        api_ok = False
    status = "OK" if api_ok else "INDISPONÍVEL"
    await interaction.followup.send(f"Pong! Latência do bot: {latency_ms}ms | API: {status}")


@bot.tree.command(name="stats", description="Mostrar estatísticas do córtex e rede sináptica")
async def slash_stats(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        async with http_session.get(f"{API_BASE_URL}/stats") as resp:
            if resp.status == 200:
                data = await resp.json()
                frag = data.get("fragment_stats", {})
                net = data.get("network_stats", {}).get("fragment_network", {})
                syn = data.get("synaptic_stats", {})
                learn = data.get("learning_stats", {})
                desc = (
                    f"Fragmentos: {frag.get('total_fragments', 0)}\n"
                    f"Arestas: {net.get('edges', 0)}\n"
                    f"Sinapses fortes: {syn.get('strong_synapses', 0)}\n"
                    f"Consultas: {learn.get('queries_processed', 0)}"
                )
                embed = discord.Embed(title="Estatísticas do Sistema", description=desc, color=0xf1c40f)
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


@bot.tree.command(name="export", description="Exportar amostra da memória")
async def slash_export(interaction: discord.Interaction, max_itens: int = 200):
    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        async with http_session.get(f"{API_BASE_URL}/export", params={"max_items": max(1, min(max_itens, 2000))}) as resp:
            if resp.status == 200:
                data = await resp.json()
                # Se pequeno, mostra embed; se grande, anexa JSON
                items = data.get("items_sample", [])
                meta = {
                    "items_total": data.get("items_total", 0),
                    "fragments_total": data.get("fragments_total", 0)
                }
                text_preview = "\n".join([f"- {i.get('id')}: {i.get('text')[:80]}" for i in items[:10]])
                embed = discord.Embed(title="Exportação de Memória", color=0x9b59b6)
                embed.add_field(name="Totais", value=f"Itens: {meta['items_total']} | Fragmentos: {meta['fragments_total']}")
                if text_preview:
                    embed.add_field(name="Amostra (10)", value=text_preview, inline=False)
                # Anexo JSON
                import json, io
                buf = io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
                file = discord.File(buf, filename="export_memoria.json")
                await interaction.followup.send(embed=embed, file=file)
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


@bot.tree.command(name="forget", description="Esquecer itens cujo texto contenha um padrão")
async def slash_forget(interaction: discord.Interaction, padrao: str, max_remover: int = 5):
    now = time.time()
    last = user_cooldowns.get(interaction.user.id, 0)
    if now - last < COOLDOWN_SECONDS:
        await interaction.response.send_message("⏳ Aguarde um pouco antes de usar o comando novamente.", ephemeral=True)
        return
    user_cooldowns[interaction.user.id] = now

    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        payload = {"pattern": padrao, "max_remove": max(1, min(max_remover, 100))}
        async with http_session.post(f"{API_BASE_URL}/forget", json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                removed = data.get("removed", 0)
                await interaction.followup.send(f"🧹 Removidos {removed} itens que combinam com '{padrao}'.")
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


@bot.tree.command(name="neo4j_sync", description="Sincronizar o grafo atual com o Neo4j")
async def slash_neo4j_sync(interaction: discord.Interaction):
    now = time.time()
    last = user_cooldowns.get(interaction.user.id, 0)
    if now - last < COOLDOWN_SECONDS:
        await interaction.response.send_message("⏳ Aguarde um pouco antes de usar o comando novamente.", ephemeral=True)
        return
    user_cooldowns[interaction.user.id] = now

    await interaction.response.defer(thinking=True)
    try:
        assert http_session is not None
        async with http_session.post(f"{API_BASE_URL}/neo4j/sync") as resp:
            if resp.status == 200:
                data = await resp.json()
                ok = data.get("ok", False)
                detail = data.get("detail", "")
                color = 0x2ecc71 if ok else 0xe74c3c
                embed = discord.Embed(title="Neo4j Sync", description=detail, color=color)
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"Erro da API: {resp.status}")
    except aiohttp.ClientError as e:
        await interaction.followup.send(f"Falha ao conectar à API: {e}")


# --- Ponto de Entrada ---
async def _main():
    if not BOT_TOKEN:
        print("ERRO: Variável de ambiente DISCORD_BOT_TOKEN não definida!")
        print("Defina-a antes de iniciar o bot. Ex.: set DISCORD_BOT_TOKEN=seu_token_no_Windows")
        return
    try:
        await bot.start(BOT_TOKEN)
    finally:
        # Fecha sessão HTTP ao encerrar
        global http_session
        if http_session is not None:
            await http_session.close()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("Bot encerrado pelo usuário.")
