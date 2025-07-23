#!/usr/bin/env python3
"""
DeployContextAgent - Agente Especializado em Deploy
==================================================

Exemplo de como usar nossa arquitetura funcional para criar
um agente especializado em deploy.

Funcionalidades:
- Processa mensagens do Slack
- Gerencia tags do Git  
- Executa deploys de backend
- Mantém contexto de deploy em YAML
- Persiste histórico de deploys
"""

import subprocess
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Importar nossa arquitetura base
from agent import FunctionalAgent, AgentConfig, AgentBuilder
from agent.tools import tool, get_tool_registry
from agent.context import ContextBuilder
from models import create_user_message, Thread, Message

# ============================================================================
# FERRAMENTAS ESPECÍFICAS DE DEPLOY
# ============================================================================

@tool(
    name="list_git_tags",
    description="Lista todas as tags disponíveis no repositório Git",
    category="git"
)
def list_git_tags() -> List[str]:
    """Lista tags do Git ordenadas por versão"""
    try:
        result = subprocess.run(
            ['git', 'tag', '-l', '--sort=-version:refname'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if result.returncode == 0:
            tags = [tag.strip() for tag in result.stdout.split('\n') if tag.strip()]
            return tags[:10]  # Últimas 10 tags
        else:
            return ["Erro: " + result.stderr]
            
    except Exception as e:
        return [f"Erro ao listar tags: {str(e)}"]

@tool(
    name="deploy_backend", 
    description="Executa deploy do backend para uma tag específica",
    category="deploy",
    requires_confirmation=True,
    is_dangerous=True
)
def deploy_backend(tag: str, environment: str = "production") -> Dict[str, Any]:
    """
    Executa deploy do backend.
    ATENÇÃO: Esta é uma simulação - substitua pela lógica real de deploy.
    """
    try:
        # Validar tag
        available_tags = list_git_tags()
        if tag not in available_tags:
            return {
                "status": "error",
                "message": f"Tag '{tag}' não encontrada. Tags disponíveis: {available_tags[:5]}"
            }
        
        # Simular deploy (substitua por lógica real)
        print(f"🚀 Iniciando deploy da tag {tag} para {environment}...")
        
        # Aqui você colocaria a lógica real de deploy:
        # - Fazer checkout da tag
        # - Build da aplicação
        # - Deploy para o ambiente
        # - Testes de smoke
        # - Rollback se necessário
        
        deploy_result = {
            "status": "success",
            "deployed_tag": tag,
            "environment": environment,
            "deployed_at": datetime.utcnow().isoformat(),
            "deployment_id": f"deploy_{datetime.utcnow().timestamp()}",
            "previous_version": available_tags[1] if len(available_tags) > 1 else None
        }
        
        print(f"✅ Deploy concluído com sucesso!")
        return deploy_result
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Erro no deploy: {str(e)}",
            "tag": tag,
            "environment": environment
        }

@tool(
    name="get_deployment_status",
    description="Verifica status do último deployment",
    category="deploy"
)
def get_deployment_status() -> Dict[str, Any]:
    """Verifica status do deployment atual"""
    # Simular verificação de status
    return {
        "current_version": "v1.2.3",
        "status": "healthy",
        "last_deployed": "2024-01-15T10:30:00Z",
        "uptime": "2 days, 5 hours",
        "health_checks": {
            "api": "healthy",
            "database": "healthy", 
            "cache": "healthy"
        }
    }

@tool(
    name="process_slack_message",
    description="Processa mensagem recebida do Slack",
    category="communication"
)
def process_slack_message(author: str, channel: str, text: str) -> Dict[str, Any]:
    """Processa e analisa mensagem do Slack"""
    
    # Detectar se é comando de deploy
    is_deploy_command = any(keyword in text.lower() for keyword in [
        'deploy', 'release', 'push', 'publish'
    ])
    
    # Extrair possível tag da mensagem
    extracted_tag = None
    words = text.split()
    for word in words:
        if word.startswith('v') and '.' in word:
            extracted_tag = word
            break
    
    return {
        "author": author,
        "channel": channel,
        "text": text,
        "processed_at": datetime.utcnow().isoformat(),
        "is_deploy_command": is_deploy_command,
        "extracted_tag": extracted_tag,
        "intent": "deploy_request" if is_deploy_command else "general_message"
    }

# ============================================================================
# CONTEXTO ESPECIALIZADO DE DEPLOY
# ============================================================================

class DeployContextBuilder(ContextBuilder):
    """
    Context builder especializado para deploy.
    Estende o builder base com informações específicas.
    """
    
    def build_deploy_system_prompt(self, thread: Thread, current_tag: str = None) -> str:
        """Constrói prompt especializado para deploy"""
        
        # Obter tags disponíveis
        try:
            available_tags = list_git_tags()[:5]  # Últimas 5 tags
        except:
            available_tags = ["v1.2.3", "v1.2.2", "v1.2.1"]
        
        # Obter status atual
        try:
            deploy_status = get_deployment_status()
        except:
            deploy_status = {"current_version": "unknown", "status": "unknown"}
        
        # Contexto especializado em YAML
        deploy_context = f"""
agent_info:
  name: "DeployContextAgent"
  version: "1.0.0"
  specialization: "Backend Deployment Automation"
  capabilities:
    - "Git tag management"
    - "Backend deployment"
    - "Slack message processing"
    - "Deploy status monitoring"
    - "Rollback operations"

current_environment:
  name: "production"
  current_version: "{deploy_status.get('current_version', 'unknown')}"
  status: "{deploy_status.get('status', 'unknown')}"
  
available_operations:
  - list_git_tags: "Lista tags disponíveis do Git"
  - deploy_backend: "Executa deploy para tag específica"
  - get_deployment_status: "Verifica status atual"
  - process_slack_message: "Processa mensagens do Slack"

recent_tags:
{chr(10).join(f"  - {tag}" for tag in available_tags)}

deployment_context:
  current_tag: "{current_tag or deploy_status.get('current_version', 'unknown')}"
  last_deployment: "{deploy_status.get('last_deployed', 'unknown')}"
  
safety_guidelines:
  - "Sempre confirmar tag antes do deploy"
  - "Verificar se tag existe no repositório"
  - "Fazer backup antes de mudanças críticas"
  - "Monitorar saúde após deploy"
  - "Ter plano de rollback pronto"

behavior_instructions: |
  Você é um agente especializado em deploy de backend. Quando receber:
  
  1. Mensagens do Slack sobre deploy:
     - Processe a mensagem para extrair informações
     - Identifique a tag solicitada
     - Confirme detalhes antes de executar
  
  2. Solicitações de deploy:
     - Liste tags disponíveis se necessário
     - Confirme a tag a ser deployada
     - Execute o deploy com segurança
     - Reporte o status final
  
  3. Verificações de status:
     - Consulte status atual do sistema
     - Reporte saúde dos componentes
     - Sugira ações se houver problemas
  
  Sempre seja cauteloso com operações de deploy e priorize a estabilidade.
"""
        
        return f"""Você é o DeployContextAgent, um assistente especializado em automação de deploy.

---
{deploy_context}
---

Responda de forma clara e técnica, sempre priorizando segurança e estabilidade nos deploys."""

# ============================================================================
# AGENTE ESPECIALIZADO EM DEPLOY
# ============================================================================

class DeployContextAgent(FunctionalAgent):
    """
    Agente especializado em deploy, baseado na arquitetura funcional.
    Implementa exatamente o fluxo da imagem fornecida.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        # Configuração específica para deploy
        if config is None:
            config = AgentConfig(
                model="openai/gpt-4",  # Modelo mais robusto para operações críticas
                temperature=0.1,  # Mais determinístico para deploy
                enable_function_calling=True,
                enable_memory_persistence=True,
                memory_db_path="deploy_memory.db",
                context_strategy="default"
            )
        
        super().__init__(config)
        
        # Context builder especializado
        self.deploy_context_builder = DeployContextBuilder()
        
        # Registrar ferramentas específicas
        self._register_deploy_tools()
    
    def _register_deploy_tools(self):
        """Registra ferramentas específicas de deploy"""
        registry = get_tool_registry()
        
        # As ferramentas já foram registradas automaticamente via decorators
        # Verificar se estão disponíveis
        tools = registry.get_available_tools()
        deploy_tools = [t for t in tools if t.get("name") in [
            "list_git_tags", "deploy_backend", "get_deployment_status", "process_slack_message"
        ]]
        
        self.logger.log_info(f"Deploy tools registradas: {len(deploy_tools)}")
    
    def process_slack_message(self, author: str, channel: str, text: str) -> Dict[str, Any]:
        """
        Processa mensagem do Slack seguindo o fluxo da imagem.
        Ponto de entrada principal para mensagens do Slack.
        """
        self.logger.log_info(
            "Processando mensagem do Slack",
            author=author,
            channel=channel,
            text=text[:100] + "..." if len(text) > 100 else text
        )
        
        # 1. Processar mensagem via tool
        slack_message = Message(
            role="user",
            content=f"Mensagem do Slack de @{author} no #{channel}: {text}"
        )
        
        # Adicionar à thread
        self.current_thread = self.current_thread.add_message(slack_message)
        
        # 2. Processar via agente
        response_result = self.process_user_message(
            f"Processe esta mensagem do Slack: author={author}, channel={channel}, text='{text}'"
        )
        
        if response_result.success:
            return {
                "status": "processed",
                "response": response_result.data,
                "session_id": self.session_id
            }
        else:
            return {
                "status": "error",
                "error": response_result.error,
                "session_id": self.session_id
            }
    
    def execute_deploy(self, tag: str, environment: str = "production") -> Dict[str, Any]:
        """
        Executa deploy seguindo o fluxo da imagem.
        """
        deploy_message = f"Executar deploy da tag {tag} para {environment}"
        
        # Processar via agente principal
        result = self.process_user_message(deploy_message)
        
        return {
            "deploy_result": result.data if result.success else None,
            "error": result.error if not result.success else None,
            "session_id": self.session_id
        }
    
    def get_deploy_context_yaml(self) -> str:
        """
        Retorna contexto atual em formato YAML.
        Implementa a parte "Context (YAML)" da imagem.
        """
        return self.deploy_context_builder.build_deploy_system_prompt(
            self.current_thread
        )

# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def main_deploy_agent():
    """
    Exemplo de uso do DeployContextAgent.
    Simula o fluxo completo da imagem.
    """
    print("🚀 Iniciando DeployContextAgent...")
    
    # Criar agente
    agent = DeployContextAgent()
    
    print(f"📍 Sessão: {agent.session_id}")
    print(f"🧠 Modelo: {agent.config.model}")
    
    # Simular mensagem do Slack
    print("\n" + "="*50)
    print("📱 Simulando mensagem do Slack...")
    
    slack_result = agent.process_slack_message(
        author="dev_team",
        channel="deploys", 
        text="Por favor, faça deploy da tag v1.2.3 para produção"
    )
    
    print(f"Resultado: {slack_result}")
    
    # Mostrar contexto YAML
    print("\n" + "="*50)
    print("📋 Contexto atual (YAML):")
    print(agent.get_deploy_context_yaml()[:500] + "...")
    
    # Interação manual
    print("\n" + "="*50)
    print("💬 Modo interativo (digite 'quit' para sair):")
    
    while True:
        try:
            user_input = input("\n👤 Você: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'sair']:
                break
            
            if not user_input:
                continue
            
            result = agent.process_user_message(user_input)
            
            if result.success:
                print(f"🤖 Deploy Agent: {result.data}")
            else:
                print(f"❌ Erro: {result.error}")
                
        except KeyboardInterrupt:
            break
    
    print("\n👋 Deploy Agent encerrado!")

if __name__ == "__main__":
    main_deploy_agent()
