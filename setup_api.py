"""ModelScope API 配置脚本"""
import os
import sys

def setup():
    print("=" * 50)
    print("  ModelScope API 配置")
    print("=" * 50)
    print()
    print("获取 API Key 的步骤：")
    print("1. 访问 https://modelscope.cn")
    print("2. 注册/登录账号")
    print("3. 点击右上角头像 → Access Token")
    print("4. 创建新的 Token 并复制")
    print()
    
    # Check if already set
    existing = os.environ.get("MODELSCOPE_API_KEY", "")
    if existing:
        print(f"当前环境变量已设置: {existing[:10]}...")
        use_existing = input("使用现有 Key? (y/n): ").strip().lower()
        if use_existing == "y":
            api_key = existing
        else:
            api_key = input("输入新的 API Key: ").strip()
    else:
        api_key = input("输入你的 ModelScope API Key: ").strip()
    
    if not api_key:
        print("未输入 Key，跳过配置。")
        return
    
    # Write .env file
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    with open(env_path, "w") as f:
        f.write(f"MODELSCOPE_API_KEY={api_key}\n")
    
    # Also set environment variable for current session
    os.environ["MODELSCOPE_API_KEY"] = api_key
    
    print()
    print(f"[OK] API Key 已保存到 {env_path}")
    print("[OK] 环境变量已设置")
    print()
    print("测试 API 连接...")
    
    import requests
    # ModelScope 是国内服务，直连（学校网络没开 Clash 也能用）
    try:
        resp = requests.post(
            "https://api-inference.modelscope.cn/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "stepfun-ai/Step-3.5-Flash",
                "messages": [{"role": "user", "content": "你好，请用一句话回答：1+1等于几？"}],
                "max_tokens": 50,
            },
            proxies={"http": None, "https": None},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        print(f"[OK] API 测试成功！回答: {answer}")
    except Exception as e:
        print(f"[WARN] API 测试失败: {e}")
        print("请检查 Key 是否正确，以及代理是否开启。")

if __name__ == "__main__":
    setup()
