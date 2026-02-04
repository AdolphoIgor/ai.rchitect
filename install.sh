#!/bin/bash

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Configuração do Ambiente AI Architect (Ubuntu 22.04) ===${NC}"

# 1. Verifica se Python 3.12 está instalado
echo -e "\n${BLUE}[1/5] Verificando Python 3.12...${NC}"
if ! command -v python3.12 &> /dev/null; then
    echo -e "${RED}Python 3.12 não encontrado.${NC}"
    echo "Instalando Python 3.12 via PPA deadsnakes..."
    
    # Adiciona repositório e instala (requer sudo)
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
else
    echo -e "${GREEN}Python 3.12 já está instalado.$(python3.12 --version)${NC}"
fi

# 2. Cria o Virtual Environment
echo -e "\n${BLUE}[2/5] Criando Virtual Environment (venv)...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}Pasta 'venv' já existe.${NC}"
else
    python3.12 -m venv .venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Venv criado com sucesso.${NC}"
    else
        echo -e "${RED}Falha ao criar venv.${NC}"
        exit 1
    fi
fi

# 3. Ativa o ambiente e atualiza o pip
echo -e "\n${BLUE}[3/5] Ativando venv e atualizando pip...${NC}"
source .venv/bin/activate
pip install -U pip

# 4. Instala as dependências
echo -e "\n${BLUE}[4/5] Instalando bibliotecas do projeto...${NC}"
# Instala google-genai (SDK nova), qdrant-client (Vector DB) e python-dotenv
pip install -U google-genai qdrant-client python-dotenv

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Dependências instaladas com sucesso!${NC}"
else
    echo -e "${RED}Erro na instalação das dependências.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=== Configuração Concluída! ===${NC}"
echo -e "Para iniciar o agente, execute:"
echo -e "${BLUE}source venv/bin/activate${NC}"
echo -e "${BLUE}python ai_architect_rag.py${NC}"