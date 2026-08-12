from crewai import Agent, Task, Crew, Process
from agents import blog_writer, blog_researcher
from task import research_task, write_task

crew = Crew(
    agents = [blog_researcher, blog_writer],
    tasks=[research_task, write_task],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm=100,
    share_crew=True

)

result = crew.kickoff(inputs={'topic':'Nemotron 3.5 Lightning vs Muse Glimmer on RTX 5090'})

print(result)