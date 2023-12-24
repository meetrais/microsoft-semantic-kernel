import semantic_kernel as sk
import apiKeys

from semantic_kernel.connectors.ai.open_ai import(
    OpenAIChatCompletion,
)
from semantic_kernel.core_skills import FileIOSkill, MathSkill, TextSkill, TimeSkill
from semantic_kernel.planning import ActionPlanner

async def main():
    kernel = sk.Kernel()
    api_key, org_id = apiKeys.OPENAI_API_KEY, apiKeys.OPENAI_ORG_ID

    kernel.add_chat_service(
        "chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key,org_id)
    )
    kernel.import_skill(MathSkill(), "math")
    kernel.import_skill(FileIOSkill(), "fileIO")
    kernel.import_skill(TimeSkill(),"time")
    kernel.import_skill(TextSkill(),"text")

    planner = ActionPlanner(kernel)

    """
    MathSkill
    ask = "What is sum of 100 and 200?"
    ChatBot: 300
    
    FileIOSkill
    ask = "What is content of file README.md?"
    ChatBot: # microsoft-semantic-kernel
    
    TimeSkill
    ask = "Whats the time  now?"
    ChatBot:  12:19:37 PM
    """
    ask = "What is sum of 100 and 900?"
    plan = await planner.create_plan_async(goal=ask)

    result = await plan.invoke_async()
    print("ChatBot: ", result)

if __name__=="__main__":
    import asyncio
    asyncio.run(main())

