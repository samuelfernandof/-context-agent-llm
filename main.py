#!/usr/bin/env python3
"""
Agente Funcional com Contexto Persistente
==========================================

Script principal para execução do agente inteligente com:
- Memória persistente (SQLite)
- Tool use via OpenAI Function Calls
- Logs estruturados
- Programação funcional e tipagem forte
- Loop autônomo com retry/fallback

Uso:
    python main.py                    # Modo interativo padrão
    python main.py --model MODEL      # Especificar modelo
    python main.py --config FILE      # Usar arquivo de configuração
    python main.py --help            # Mostrar ajuda completa

Autor: Agente Funcional v1.0
"""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

# Importações do projeto
from agent.agent import (
    FunctionalAgent, AgentConfig, AgentBuilder,
    quick_start, create_production_agent, create_development_agent,
    validate_environment, load_config_from_env
)
from agent.logger import get_logger
from agent.memory import get_memory, backup_memory
from models.models import Result

# ============================================================================
# CONFIGURAÇÃO E ARGUMENTOS DA LINHA DE COMANDO
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Cria parser para argumentos da linha de comando.
    Função pura de configuração.
    """
    parser = argparse.ArgumentParser(
        description="Agente Funcional com Contexto Persistente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py                              # Modo interativo básico
  python main.py --model openai/gpt-4         # Usar GPT-4
  python main.py --temperature 0.3            # Baixa criatividade
  python main.py --no-memory                  # Desabilitar persistência
  python main.py --dev                        # Modo desenvolvimento
  python main.py --prod                       # Modo produção
  python main.py --config config.json         # Carregar configuração
  python main.py --export-config              # Exportar configuração padrão
  python main.py --backup-memory              # Fazer backup da memória
  python main.py --stats                      # Mostrar estatísticas

Variáveis de ambiente suportadas:
  OPENROUTER_API_KEY      # Chave da API (obrigatória)
  AGENT_MODEL            # Modelo LLM padrão
  AGENT_TEMPERATURE      # Temperatura do modelo (0-2)
  AGENT_MAX_TOKENS       # Máximo de tokens por resposta
  AGENT_MEMORY_DB        # Caminho do banco de memória
  AGENT_CONTEXT_STRATEGY # Estratégia de contexto
  AGENT_MAX_RETRIES      # Máximo de tentativas
  AGENT_ENABLE_TOOLS     # Habilitar ferramentas (true/false)
        """
    )
    
    # Argumentos principais
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Modelo LLM a usar (ex: openai/gpt-4, mistralai/mistral-7b-instruct)"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Chave da API OpenRouter (sobrescreve variável de ambiente)"
    )
    
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        help="Temperatura do modelo (0.0-2.0, padrão: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Máximo de tokens por resposta (padrão: 2000)"
    )
    
    parser.add_argument(
        "--session-id",
        type=str,
        help="ID da sessão específica para carregar/continuar"
    )
    
    # Configurações de memória
    parser.add_argument(
        "--memory-db",
        type=str,
        help="Caminho para o banco de dados de memória"
    )
    
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Desabilitar persistência de memória"
    )
    
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Desabilitar execução de ferramentas"
    )
    
    # Estratégias de contexto
    parser.add_argument(
        "--context-strategy",
        choices=["default", "recent_only", "compressed", "minimal"],
        help="Estratégia de gerenciamento de contexto"
    )
    
    # Configurações de retry
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Máximo de tentativas em caso de erro"
    )
    
    # Modos pré-configurados
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Modo desenvolvimento (configurações otimizadas para dev)"
    )
    
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Modo produção (configurações otimizadas e robustas)"
    )
    
    # Configuração via arquivo
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Carregar configuração de arquivo JSON"
    )
    
    parser.add_argument(
        "--export-config",
        type=str,
        nargs="?",
        const="agent_config.json",
        help="Exportar configuração padrão para arquivo"
    )
    
    # Utilitários
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostrar estatísticas da memória e sair"
    )
    
    parser.add_argument(
        "--backup-memory",
        type=str,
        nargs="?",
        const=None,
        help="Fazer backup da memória para arquivo específico"
    )
    
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="Listar sessões disponíveis na memória"
    )
    
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validar ambiente e dependências"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Saída verbosa com logs detalhados"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Saída mínima (apenas erros)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Agente Funcional v1.0.0"
    )
    
    return parser

def load_config_from_file(config_path: str) -> Result:
    """
    Carrega configuração de arquivo JSON.
    Função pura de I/O com tratamento de erros.
    """
    try:
        config_file = Path(config_path)
        
        if not config_file.exists():
            return Result.error(f"Arquivo de configuração não encontrado: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Validar campos obrigatórios
        valid_fields = set(AgentConfig.__dataclass_fields__.keys())
        config_fields = set(config_data.keys())
        
        invalid_fields = config_fields - valid_fields
        if invalid_fields:
            return Result.error(f"Campos inválidos na configuração: {', '.join(invalid_fields)}")
        
        # Criar configuração
        config = AgentConfig(**config_data)
        return Result.ok(config)
        
    except json.JSONDecodeError as e:
        return Result.error(f"Erro ao parsear JSON: {str(e)}")
    except Exception as e:
        return Result.error(f"Erro ao carregar configuração: {str(e)}")

def save_config_to_file(config: AgentConfig, config_path: str) -> Result:
    """
    Salva configuração em arquivo JSON.
    Função com efeito colateral controlado.
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Converter para dicionário serializável
        config_dict = {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        return Result.ok(f"Configuração salva em: {config_path}")
        
    except Exception as e:
        return Result.error(f"Erro ao salvar configuração: {str(e)}")

def build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    """
    Constrói configuração a partir dos argumentos da linha de comando.
    Função pura de transformação.
    """
    # Começar com configuração padrão
    config = load_config_from_env()
    
    # Sobrescrever com argumentos fornecidos
    if args.model:
        config = AgentConfig(**{**config.__dict__, 'model': args.model})
    
    if args.api_key:
        config = AgentConfig(**{**config.__dict__, 'api_key': args.api_key})
    
    if args.temperature is not None:
        config = AgentConfig(**{**config.__dict__, 'temperature': args.temperature})
    
    if args.max_tokens:
        config = AgentConfig(**{**config.__dict__, 'max_tokens': args.max_tokens})
    
    if args.memory_db:
        config = AgentConfig(**{**config.__dict__, 'memory_db_path': args.memory_db})
    
    if args.no_memory:
        config = AgentConfig(**{**config.__dict__, 'enable_memory_persistence': False})
    
    if args.no_tools:
        config = AgentConfig(**{**config.__dict__, 'enable_function_calling': False})
    
    if args.context_strategy:
        config = AgentConfig(**{**config.__dict__, 'context_strategy': args.context_strategy})
    
    if args.max_retries:
        config = AgentConfig(**{**config.__dict__, 'max_retries': args.max_retries})
    
    return config

# ============================================================================
# COMANDOS UTILITÁRIOS
# ============================================================================

def show_stats(args: argparse.Namespace) -> int:
    """
    Mostra estatísticas da memória.
    Função utilitária com saída formatada.
    """
    try:
        memory = get_memory(args.memory_db or "memory.db")
        stats_result = memory.get_stats()
        
        if not stats_result.success:
            print(f"❌ Erro ao obter estatísticas: {stats_result.error}")
            return 1
        
        stats = stats_result.data
        
        print("📊 Estatísticas da Memória:")
        print(f"  📁 Banco de dados: {stats['db_path']}")
        print(f"  💽 Tamanho: {stats['db_size_mb']} MB")
        print(f"  🗂️  Total de threads: {stats['total_threads']}")
        print(f"  💬 Total de mensagens: {stats['total_messages']}")
        print(f"  📝 Total de eventos: {stats['total_events']}")
        print(f"  📅 Última atividade: {stats.get('latest_activity', 'N/A')}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Erro ao mostrar estatísticas: {str(e)}")
        return 1

def backup_memory_command(args: argparse.Namespace) -> int:
    """
    Executa backup da memória.
    Função utilitária com tratamento de erros.
    """
    try:
        source_db = args.memory_db or "memory.db"
        backup_path = args.backup_memory
        
        print(f"🔄 Fazendo backup de {source_db}...")
        
        backup_result = backup_memory(source_db, backup_path)
        
        if backup_result.success:
            data = backup_result.data
            print(f"✅ Backup concluído com sucesso!")
            print(f"  📂 Origem: {data['source']}")
            print(f"  💾 Backup: {data['backup']}")
            print(f"  📏 Tamanho: {data['size_mb']} MB")
            return 0
        else:
            print(f"❌ Erro no backup: {backup_result.error}")
            return 1
            
    except Exception as e:
        print(f"❌ Erro inesperado no backup: {str(e)}")
        return 1

def list_sessions_command(args: argparse.Namespace) -> int:
    """
    Lista sessões disponíveis na memória.
    Função utilitária com saída formatada.
    """
    try:
        memory = get_memory(args.memory_db or "memory.db")
        sessions_result = memory.list_sessions(limit=20)
        
        if not sessions_result.success:
            print(f"❌ Erro ao listar sessões: {sessions_result.error}")
            return 1
        
        sessions = sessions_result.data
        
        if not sessions:
            print("📭 Nenhuma sessão encontrada na memória.")
            return 0
        
        print(f"📋 Sessões Disponíveis ({len(sessions)}):")
        print()
        
        for i, session in enumerate(sessions, 1):
            created = session['created_at'][:19].replace('T', ' ')  # Formato legível
            updated = session['updated_at'][:19].replace('T', ' ')
            
            print(f"  {i:2d}. 🆔 {session['session_id']}")
            print(f"      📅 Criada: {created}")
            print(f"      🔄 Atualizada: {updated}")
            print(f"      💬 Mensagens: {session['message_count']}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Erro ao listar sessões: {str(e)}")
        return 1

def validate_environment_command(args: argparse.Namespace) -> int:
    """
    Valida ambiente e dependências.
    Função utilitária de diagnóstico.
    """
    try:
        print("🔍 Validando ambiente...")
        
        validation_result = validate_environment()
        
        if validation_result.success:
            data = validation_result.data
            
            print("✅ Validação do ambiente concluída com sucesso!")
            
            if data.get("warnings"):
                print("\n⚠️  Avisos encontrados:")
                for warning in data["warnings"]:
                    print(f"  • {warning}")
            else:
                print("🎉 Nenhum problema detectado!")
            
            return 0
        else:
            data = validation_result.data
            
            print("❌ Problemas encontrados no ambiente:")
            for error in data["errors"]:
                print(f"  • {error}")
            
            if data.get("warnings"):
                print("\n⚠️  Avisos adicionais:")
                for warning in data["warnings"]:
                    print(f"  • {warning}")
            
            return 1
            
    except Exception as e:
        print(f"❌ Erro na validação: {str(e)}")
        return 1

def export_config_command(args: argparse.Namespace) -> int:
    """
    Exporta configuração padrão para arquivo.
    Função utilitária de configuração.
    """
    try:
        config = AgentConfig()
        config_path = args.export_config
        
        print(f"📤 Exportando configuração padrão para: {config_path}")
        
        save_result = save_config_to_file(config, config_path)
        
        if save_result.success:
            print(f"✅ {save_result.data}")
            print("\n💡 Você pode editar este arquivo e usar com --config")
            return 0
        else:
            print(f"❌ {save_result.error}")
            return 1
            
    except Exception as e:
        print(f"❌ Erro ao exportar configuração: {str(e)}")
        return 1

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main() -> int:
    """
    Função principal do script.
    Processa argumentos e executa ação apropriada.
    """
    try:
        # Parsear argumentos
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Configurar nível de logging baseado em verbose/quiet
        logger = get_logger()
        
        # Comandos utilitários (executam e saem)
        if args.stats:
            return show_stats(args)
        
        if args.backup_memory is not None:
            return backup_memory_command(args)
        
        if args.list_sessions:
            return list_sessions_command(args)
        
        if args.validate_env:
            return validate_environment_command(args)
        
        if args.export_config:
            return export_config_command(args)
        
        # Validar ambiente antes de continuar
        if not args.quiet:
            print("🔍 Validando ambiente...")
        
        validation_result = validate_environment()
        if not validation_result.success:
            if not args.quiet:
                print("❌ Problemas no ambiente:")
                for error in validation_result.data["errors"]:
                    print(f"  • {error}")
            return 1
        
        # Mostrar warnings se não estiver em modo quiet
        if not args.quiet and validation_result.data.get("warnings"):
            print("⚠️  Avisos:")
            for warning in validation_result.data["warnings"]:
                print(f"  • {warning}")
            print()
        
        # Carregar configuração
        config = None
        
        if args.config:
            # Carregar de arquivo
            if not args.quiet:
                print(f"📋 Carregando configuração de: {args.config}")
            
            config_result = load_config_from_file(args.config)
            if not config_result.success:
                print(f"❌ Erro ao carregar configuração: {config_result.error}")
                return 1
            
            config = config_result.data
        else:
            # Construir a partir dos argumentos
            config = build_config_from_args(args)
        
        # Validar configuração
        config_validation = config.validate()
        if not config_validation.success:
            print(f"❌ Configuração inválida: {config_validation.error}")
            return 1
        
        # Criar agente baseado no modo
        agent = None
        
        if args.dev:
            if not args.quiet:
                print("🔧 Criando agente em modo desenvolvimento...")
            agent = create_development_agent(args.session_id)
            
        elif args.prod:
            if not args.quiet:
                print("🏭 Criando agente em modo produção...")
            agent = create_production_agent(args.session_id)
            
        else:
            # Criar agente com configuração personalizada
            if not args.quiet:
                print("🤖 Criando agente personalizado...")
            
            agent = FunctionalAgent(config, args.session_id)
        
        # Log inicial se verbose
        if args.verbose:
            logger.log_info(
                "Agente criado via linha de comando",
                model=config.model,
                session_id=agent.session_id,
                memory_enabled=config.enable_memory_persistence,
                tools_enabled=config.enable_function_calling
            )
        
        # Iniciar conversa
        if not args.quiet:
            print("🚀 Iniciando agente...")
            print()
        
        agent.start_conversation()
        
        return 0
        
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n🛑 Interrompido pelo usuário.")
        return 130  # Código padrão para SIGINT
        
    except Exception as e:
        print(f"❌ Erro fatal: {str(e)}")
        
        # Log do erro se possível
        try:
            get_logger().log_error(f"Erro fatal no main: {str(e)}")
        except:
            pass  # Falha silenciosa no logging
        
        return 1

# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA PARA IMPORTAÇÃO
# ============================================================================

def create_agent_from_args(args: list = None) -> FunctionalAgent:
    """
    Cria agente a partir de argumentos da linha de comando.
    Útil para uso programático.
    
    Args:
        args: Lista de argumentos (padrão: sys.argv[1:])
    
    Returns:
        FunctionalAgent configurado
    """
    if args is None:
        args = sys.argv[1:]
    
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)
    
    # Validar ambiente
    validation_result = validate_environment()
    if not validation_result.success:
        raise RuntimeError(f"Ambiente inválido: {validation_result.data['errors']}")
    
    # Construir configuração
    if parsed_args.config:
        config_result = load_config_from_file(parsed_args.config)
        if not config_result.success:
            raise ValueError(f"Erro na configuração: {config_result.error}")
        config = config_result.data
    else:
        config = build_config_from_args(parsed_args)
    
    # Validar configuração
    config_validation = config.validate()
    if not config_validation.success:
        raise ValueError(f"Configuração inválida: {config_validation.error}")
    
    # Criar agente apropriado
    if parsed_args.dev:
        return create_development_agent(parsed_args.session_id)
    elif parsed_args.prod:
        return create_production_agent(parsed_args.session_id)
    else:
        return FunctionalAgent(config, parsed_args.session_id)

def run_agent_with_config(config_path: str, session_id: Optional[str] = None) -> None:
    """
    Executa agente com arquivo de configuração.
    Função de conveniência para uso programático.
    """
    # Carregar configuração
    config_result = load_config_from_file(config_path)
    if not config_result.success:
        raise ValueError(f"Erro ao carregar configuração: {config_result.error}")
    
    config = config_result.data
    
    # Validar configuração
    config_validation = config.validate()
    if not config_validation.success:
        raise ValueError(f"Configuração inválida: {config_validation.error}")
    
    # Criar e executar agente
    agent = FunctionalAgent(config, session_id)
    agent.start_conversation()

def interactive_setup() -> AgentConfig:
    """
    Setup interativo para configuração do agente.
    Útil para usuários iniciantes.
    """
    print("🎯 Setup Interativo do Agente Funcional")
    print("=" * 40)
    print()
    
    # API Key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("🔑 API Key não encontrada no ambiente.")
        api_key = input("Digite sua chave da API OpenRouter: ").strip()
        if not api_key:
            raise ValueError("API Key é obrigatória")
    else:
        print(f"✅ API Key encontrada: {api_key[:8]}...")
    
    # Modelo
    print("\n🧠 Modelos disponíveis:")
    models = [
        ("mistralai/mistral-7b-instruct", "Mistral 7B (rápido, econômico)"),
        ("openai/gpt-3.5-turbo", "GPT-3.5 Turbo (balanceado)"),
        ("openai/gpt-4", "GPT-4 (melhor qualidade, mais caro)"),
        ("anthropic/claude-3-sonnet", "Claude 3 Sonnet (criativo)"),
    ]
    
    for i, (model, desc) in enumerate(models, 1):
        print(f"  {i}. {model} - {desc}")
    
    while True:
        try:
            choice = input(f"\nEscolha o modelo (1-{len(models)}) [1]: ").strip()
            if not choice:
                choice = "1"
            
            model_index = int(choice) - 1
            if 0 <= model_index < len(models):
                selected_model = models[model_index][0]
                break
            else:
                print("❌ Opção inválida!")
        except ValueError:
            print("❌ Digite um número válido!")
    
    # Temperatura
    while True:
        try:
            temp_input = input("\n🌡️  Temperature (0.0-2.0) [0.7]: ").strip()
            if not temp_input:
                temperature = 0.7
                break
            
            temperature = float(temp_input)
            if 0.0 <= temperature <= 2.0:
                break
            else:
                print("❌ Temperature deve estar entre 0.0 e 2.0!")
        except ValueError:
            print("❌ Digite um número válido!")
    
    # Memória
    enable_memory = input("\n💾 Habilitar memória persistente? (S/n) [S]: ").strip().lower()
    enable_memory = enable_memory != 'n'
    
    # Ferramentas
    enable_tools = input("\n🛠️  Habilitar ferramentas/funções? (S/n) [S]: ").strip().lower()
    enable_tools = enable_tools != 'n'
    
    # Criar configuração
    config = AgentConfig(
        model=selected_model,
        api_key=api_key,
        temperature=temperature,
        enable_memory_persistence=enable_memory,
        enable_function_calling=enable_tools
    )
    
    print("\n✅ Configuração criada com sucesso!")
    print(f"   🧠 Modelo: {config.model}")
    print(f"   🌡️  Temperature: {config.temperature}")
    print(f"   💾 Memória: {'✅' if config.enable_memory_persistence else '❌'}")
    print(f"   🛠️  Ferramentas: {'✅' if config.enable_function_calling else '❌'}")
    
    # Salvar configuração
    save_config = input("\n💾 Salvar configuração em arquivo? (s/N) [N]: ").strip().lower()
    if save_config in ['s', 'sim', 'yes', 'y']:
        config_path = input("Nome do arquivo [agent_config.json]: ").strip()
        if not config_path:
            config_path = "agent_config.json"
        
        save_result = save_config_to_file(config, config_path)
        if save_result.success:
            print(f"✅ Configuração salva em: {config_path}")
        else:
            print(f"❌ Erro ao salvar: {save_result.error}")
    
    return config

# ============================================================================
# COMPATIBILIDADE COM CÓDIGO LEGADO
# ============================================================================

def main_agent():
    """
    Função de compatibilidade com o código legado.
    Redireciona para a função principal.
    """
    return main()

# ============================================================================
# TRATAMENTO DE SINAIS (UNIX)
# ============================================================================

def setup_signal_handlers():
    """
    Configura handlers para sinais do sistema (UNIX only).
    Permite encerramento gracioso.
    """
    import signal
    
    def signal_handler(signum, frame):
        print(f"\n🛑 Sinal {signum} recebido. Encerrando graciosamente...")
        sys.exit(0)
    
    # Capturar sinais comuns
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill
    
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)   # Terminal fechado

# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Configurar handlers de sinal em sistemas Unix
    try:
        setup_signal_handlers()
    except:
        pass  # Ignorar em Windows ou outros sistemas
    
    # Executar função principal
    exit_code = main()
    sys.exit(exit_code)
