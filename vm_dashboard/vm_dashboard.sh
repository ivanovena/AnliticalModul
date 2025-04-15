#!/bin/bash

# Script para iniciar y manejar la máquina virtual del dashboard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    echo "Script para manejar la máquina virtual del dashboard con Ubuntu GUI"
    echo ""
    echo "Uso: ./vm_dashboard.sh [comando]"
    echo ""
    echo "Comandos:"
    echo "  start       Inicia la máquina virtual (la crea si no existe)"
    echo "  stop        Detiene la máquina virtual"
    echo "  destroy     Elimina la máquina virtual"
    echo "  status      Muestra el estado de la máquina virtual"
    echo "  ssh         Conectarse a la máquina virtual mediante SSH"
    echo "  help        Muestra esta ayuda"
    echo ""
    echo "Nota: Necesitas tener instalados VirtualBox y Vagrant en tu sistema."
}

check_requirements() {
    if ! command -v vagrant &> /dev/null; then
        echo "Error: Vagrant no está instalado."
        echo "Por favor, instala Vagrant desde https://www.vagrantup.com/downloads"
        exit 1
    fi
    
    if ! command -v VBoxManage &> /dev/null; then
        echo "Error: VirtualBox no está instalado."
        echo "Por favor, instala VirtualBox desde https://www.virtualbox.org/wiki/Downloads"
        exit 1
    fi
}

start_vm() {
    echo "Iniciando la máquina virtual del dashboard..."
    vagrant up
    echo ""
    echo "La máquina virtual está lista. Se ha abierto una ventana con Ubuntu GUI."
    echo "Puedes acceder al dashboard desde la VM o en tu navegador local: http://localhost:3001"
    echo "Usuario: vagrant"
    echo "Contraseña: vagrant"
}

stop_vm() {
    echo "Deteniendo la máquina virtual..."
    vagrant halt
}

destroy_vm() {
    echo "¿Estás seguro de que quieres eliminar la máquina virtual? (s/n)"
    read -r response
    if [[ "$response" =~ ^([sS][iI]|[sS])$ ]]; then
        echo "Eliminando la máquina virtual..."
        vagrant destroy -f
    else
        echo "Operación cancelada."
    fi
}

status_vm() {
    vagrant status
}

ssh_vm() {
    vagrant ssh
}

# Verificar requisitos
check_requirements

# Procesar comandos
case "$1" in
    start)
        start_vm
        ;;
    stop)
        stop_vm
        ;;
    destroy)
        destroy_vm
        ;;
    status)
        status_vm
        ;;
    ssh)
        ssh_vm
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Comando no reconocido: $1"
        show_help
        exit 1
        ;;
esac

exit 0