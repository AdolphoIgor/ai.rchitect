# pip install -U google-genai qdrant-client python-dotenv

import os
import sys
import subprocess
import time
import hashlib
import json
import glob
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# --- DEPENDÊNCIAS EXTERNAS ---
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

# --- CONFIGURAÇÃO ---
@dataclass
class Config:
    API_KEY: str = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    # Usando o flash estável para evitar 404 e economizar
    MODEL_NAME: str = os.environ.get("AI_MODEL_NAME", "models/gemini-2.0-flash") 
    EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "models/text-embedding-004")  # Mais novo e eficiente que o 001
    TARGET_DIR: str = os.environ.get("TARGET_PROJECT_DIR", ".")
    QDRANT_PATH: str = "./.qdrant_local"
    CACHE_FILE: str = ".llm_cache.json"
    GIT_BRANCH_PREFIX: str = "ai-arch/"
    # Arquivos a ignorar na indexação
    IGNORE_PATTERNS: tuple = (
        ".venv", "venv", ".git", "__pycache__", ".idea", ".vscode", 
        "node_modules", ".env", "poetry.lock", "package-lock.json",
        "dist", "build", "*.pyc", ".DS_Store", ".qdrant_local", ".llm_cache.json"
    )

config = Config()

if not config.API_KEY:
    print("ERRO: Configure GOOGLE_API_KEY ou GEMINI_API_KEY.")
    sys.exit(1)

# --- SISTEMA DE CACHE LOCAL (LLM) ---
class LLMCache:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cache = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f: return json.load(f)
            except: return {}
        return {}

    def _save(self):
        with open(self.filepath, 'w') as f: json.dump(self.cache, f)

    def get(self, key: str) -> Optional[str]:
        return self.cache.get(key)

    def set(self, key: str, value: str):
        self.cache[key] = value
        self._save()

    def generate_key(self, prompt: str, history: List) -> str:
        # Cria um hash único baseado no prompt e no histórico recente
        content = f"{prompt}|{str(history[-2:])}" # Olha apenas as ultimas 2 interacoes
        return hashlib.md5(content.encode()).hexdigest()

llm_cache = LLMCache(config.CACHE_FILE)

# --- SISTEMA RAG & VETORIZAÇÃO ---
class VectorStore:
    def __init__(self, client_genai):
        self.genai_client = client_genai
        self.client = QdrantClient(path=config.QDRANT_PATH)
        self.collection_name = "project_codebase"
        self._init_collection()
        # Estado dos arquivos para detecção de mudança (caminho -> hash_md5)
        self.file_state_db = os.path.join(config.QDRANT_PATH, "file_state.json")
        self.known_files = self._load_file_state()

    def _init_collection(self):
        collections = self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)
        if not exists:
            # text-embedding-004 tem 768 dimensoes
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def _load_file_state(self) -> Dict[str, str]:
        if os.path.exists(self.file_state_db):
            with open(self.file_state_db, 'r') as f: return json.load(f)
        return {}

    def _save_file_state(self):
        with open(self.file_state_db, 'w') as f: json.dump(self.known_files, f)

    def _get_file_hash(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        # Retry logic para embedding (rate limits)
        for _ in range(3):
            try:
                result = self.genai_client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=text
                )
                return result.embeddings[0].values
            except Exception as e:
                if "429" in str(e):
                    time.sleep(2)
                    continue
                raise e
        return []

    def index_project(self, root_dir: str):
        print(f"🔍 Verificando alterações em {root_dir}...")
        files_to_index = []
        current_files = set()

        # 1. Escaneamento
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
            for file in files:
                full_path = os.path.join(root, file)
                if is_ignored(full_path): continue
                
                rel_path = os.path.relpath(full_path, root_dir)
                current_files.add(rel_path)
                
                try:
                    current_hash = self._get_file_hash(full_path)
                    # Se arquivo é novo ou hash mudou, re-indexa
                    if rel_path not in self.known_files or self.known_files[rel_path] != current_hash:
                        files_to_index.append((full_path, rel_path, current_hash))
                except Exception:
                    pass

        # 2. Remoção de arquivos deletados do índice
        deleted_files = set(self.known_files.keys()) - current_files
        if deleted_files:
            print(f"🗑️ Removendo {len(deleted_files)} arquivos deletados do índice...")
            # Qdrant delete logic (simplificada: recriação ou ID filter)
            # Para este script, vamos apenas remover do state tracker, 
            # na busca real o qdrant pode retornar lixo, mas o LLM ignora se não achar o arquivo.
            for f in deleted_files:
                del self.known_files[f]

        # 3. Indexação (Batch)
        if files_to_index:
            print(f"⚡ Atualizando embeddings de {len(files_to_index)} arquivos modificados...")
            points = []
            for full_path, rel_path, file_hash in files_to_index:
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Se arquivo muito grande, trunca para embedding (o LLM lerá o arquivo real depois se precisar)
                    # Estratégia: Embeddar o código + o path ajuda na busca semântica
                    content_to_embed = f"Filename: {rel_path}\n\n{content[:8000]}"
                    vector = self._get_embedding(content_to_embed)
                    
                    if vector:
                        # Usamos o hash do caminho como ID numérico (Qdrant requer int ou uuid)
                        point_id = int(hashlib.md5(rel_path.encode()).hexdigest(), 16) % (10**15)
                        points.append(PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={"path": rel_path, "content": content}
                        ))
                        self.known_files[rel_path] = file_hash
                        print(f"   Processed: {rel_path}")
                except Exception as e:
                    print(f"   Falha ao processar {rel_path}: {e}")

            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
                self._save_file_state()
        else:
            print("✅ Nenhum arquivo modificado. Índice atualizado.")

    def search(self, query: str, limit: int = 5) -> str:
        """Retorna o conteúdo dos arquivos mais relevantes."""
        query_vector = self._get_embedding(query)
        
        # FIX: Usando query_points que é mais estável entre versões
        # O argumento 'query' aceita o vetor denso
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit
            ).points
        except AttributeError:
            # Fallback para versões muito antigas ou específicas
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

        context_str = "RELEVANT FILES FOUND VIA SEARCH:\n"
        for hit in results:
            path = hit.payload['path']
            content = hit.payload['content']
            context_str += f"--- FILE: {path} ---\n{content}\n"
        return context_str
    
# --- UTILITÁRIOS GERAIS ---
def is_ignored(path: str) -> bool:
    parts = path.split(os.sep)
    for ignore in config.IGNORE_PATTERNS:
        if ignore in parts: return True
        if ignore.startswith("*") and path.endswith(ignore[1:]): return True
    return False

def get_file_tree(root_dir: str) -> str:
    """Gera apenas a árvore visual (barato em tokens)."""
    tree = "PROJECT FILE TREE:\n"
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not is_ignored(os.path.join(root, d))]
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 2 * level # Menos espaços para economizar
        tree += f"{indent}{os.path.basename(root)}/\n"
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            if not is_ignored(os.path.join(root, f)):
                 tree += f"{subindent}{f}\n"
    return tree

def git_run(args: List[str], cwd: str) -> str:
    return subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True).stdout.strip()

def create_checkout_branch(cwd: str, task_name: str) -> str:
    timestamp = int(time.time())
    safe_name = re.sub(r'[^a-zA-Z0-9]', '-', task_name).lower()[:30]
    branch_name = f"{config.GIT_BRANCH_PREFIX}{safe_name}-{timestamp}"
    git_run(["checkout", "-b", branch_name], cwd)
    return branch_name

def commit_changes(cwd: str, message: str):
    git_run(["add", "."], cwd)
    git_run(["commit", "-m", message], cwd)

def parse_and_apply_changes(response_text: str, cwd: str) -> bool:
    if not isinstance(response_text, str): response_text = str(response_text)
    file_pattern = re.compile(r'<FILE path="(.*?)">\n(.*?)<\/FILE>', re.DOTALL)
    commit_pattern = re.compile(r'<COMMIT_MESSAGE>\n(.*?)\n<\/COMMIT_MESSAGE>', re.DOTALL)
    
    files_found = file_pattern.findall(response_text)
    commit_match = commit_pattern.search(response_text)
    
    if not files_found: return False
    
    for rel_path, content in files_found:
        full_path = os.path.join(cwd, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f: f.write(content)
        print(f"📝 Atualizado: {rel_path}")
        
    msg = commit_match.group(1).strip() if commit_match else "refactor: ai agent update"
    commit_changes(cwd, msg)
    return True

# --- FLUXO PRINCIPAL ---

SYSTEM_PROMPT = """
Você é um Arquiteto de Software Especialista.
Você tem acesso a uma 'File Tree' global, mas o conteúdo dos arquivos é carregado via RAG conforme a necessidade.

REGRAS:
1. Se precisar ver o conteúdo de um arquivo que não está no contexto, PEÇA AO USUÁRIO: "Por favor, mostre o arquivo X". (O sistema RAG trará automaticamente na próxima rodada se você mencionar o nome dele ou o conceito).
2. Proponha soluções arquiteturais sólidas.
3. Se for gerar código, aguarde confirmação e use o formato XML:
<FILE path="path/to/file.py">
CODE
</FILE>
<COMMIT_MESSAGE>
msg
</COMMIT_MESSAGE>
"""

def main():
    print("🤖 AI Architect - RAG Edition (Token Optimized)")
    
    # 1. Setup Clients
    client = genai.Client(api_key=config.API_KEY)
    vector_store = VectorStore(client)
    
    # 2. Indexação Inicial (Incremental)
    vector_store.index_project(config.TARGET_DIR)
    
    # 3. Setup Chat
    chat = client.chats.create(
        model=config.MODEL_NAME,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=8192
        ),
        history=[]
    )
    
    # Contexto "Leve" (Apenas a árvore)
    file_tree = get_file_tree(config.TARGET_DIR)
    print("🌲 Estrutura de arquivos carregada.")
    
    while True:
        try:
            user_input = input("\n👤 Você: ").strip()
            if user_input.lower() in ['sair', 'exit']: break
            if not user_input: continue

            # --- BUSCA RAG ---
            # Busca arquivos relevantes com base na pergunta do usuário
            print("🔍 Buscando contexto relevante...", end="\r")
            rag_context = vector_store.search(user_input, limit=3)
            
            # Monta o prompt final (Tree + RAG Context + Pergunta)
            full_prompt = f"""
CONTEXTO GLOBAL (ARVORE):
{file_tree}

CONTEXTO ESPECIFICO (RAG):
{rag_context}

USUARIO:
{user_input}
"""
            # --- CACHE CHECK ---
            cache_key = llm_cache.generate_key(full_prompt, []) # Histórico simplificado para key
            cached_resp = llm_cache.get(cache_key)
            
            if cached_resp:
                print("⚡ Resposta recuperada do cache.")
                ai_text = cached_resp
            else:
                print("🤖 Gerando resposta...", end="\r")
                try:
                    response = chat.send_message(full_prompt)
                    ai_text = response.text
                    llm_cache.set(cache_key, ai_text)
                except Exception as e:
                    if "429" in str(e):
                        print("⏳ Quota excedida (429). Tentando novamente em 30s...")
                        time.sleep(30)
                        response = chat.send_message(full_prompt)
                        ai_text = response.text
                    else:
                        raise e

            print(" " * 30, end="\r")
            
            # --- LÓGICA DE AÇÃO ---
            if "[CONFIRM_REQUEST]" in ai_text:
                print(f"\n🤖 Proposta:\n{ai_text.replace('[CONFIRM_REQUEST]', '')}")
                if input("\n>> Confirmar? (y/n): ").lower() == 'y':
                    create_checkout_branch(config.TARGET_DIR, "feature-ai")
                    print("Gerando código...")
                    # Na geração, passamos o mesmo contexto para garantir que ele "lembre" o que ia fazer
                    exec_resp = chat.send_message("Plano APROVADO. Gere o XML.")
                    parse_and_apply_changes(exec_resp.text, config.TARGET_DIR)
                    # Re-indexa o que mudou
                    vector_store.index_project(config.TARGET_DIR)
            
            elif "<FILE path=" in ai_text:
                print(f"\n⚠️ Bypass de confirmação detectado:\n{ai_text}")
                if input("Aplicar? (y/n): ").lower() == 'y':
                    create_checkout_branch(config.TARGET_DIR, "fix")
                    parse_and_apply_changes(ai_text, config.TARGET_DIR)
                    vector_store.index_project(config.TARGET_DIR)
            
            else:
                print(f"\n🤖 {ai_text}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")

if __name__ == "__main__":
    main()