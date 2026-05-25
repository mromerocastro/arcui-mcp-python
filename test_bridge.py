import asyncio
from arcui_mcp.bridge import bridge

async def main():
    print("Intentando conectar con ArcUI en Unity (http://localhost:17842)...")
    try:
        health = await bridge.get_system_health()
        print("\n[SUCCESS] Conexion exitosa!")
        print("Salud del sistema:", health)
        
        tags = await bridge.list_tags()
        print("\n[SUCCESS] Tags disponibles:")
        print(tags)
        
    except Exception as e:
        print("\n[ERROR] Error de conexion:")
        print(e)
        print("\nPor favor, asegúrate de que Unity está abierto y le has dado al botón 'Play'.")

if __name__ == "__main__":
    asyncio.run(main())
