# Forked from my colab notebook

# Setup
# pip install langchain-google-vertexai google-cloud-aiplatform langchain langchain-community praw

# Authenticate user in colab
from google.colab import auth
auth.authenticate_user()
print('Authenticated!')

# Reddit tools
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_community.utilities.reddit_search import RedditSearchAPIWrapper
from langchain_community.document_loaders import RedditPostsLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Type
from typing import Dict, List, Any
import praw
from langchain_core.tools import Tool


# reddit API secrets
client_id = "<redacted>"
client_secret = "<redacted>"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:135.0) Gecko/20100101 Firefox/135.0"


class RedditSearchAPIWrapperCustom(RedditSearchAPIWrapper):
    def run(self, query: str, sort: str, time_filter: str, subreddit: str, limit: int) -> str:
        """Search Reddit and return posts as a single string."""
        results: List[Dict] = self.results(
            query=query,
            sort=sort,
            time_filter=time_filter,
            subreddit=subreddit,
            limit=limit,
        )
        if len(results) > 0:
            output: List[str] = [f"Searching r/{subreddit} found {len(results)} posts:"]
            for r in results:
                category = "N/A" if r["post_category"] is None else r["post_category"]
                p = f"Post Title: '{r['post_title']}'\n\
                    User: {r['post_author']}\n\
                    Subreddit: {r['post_subreddit']}:\n\
                    Text body: {r['post_text']}\n\
                    Post ID: {r['post_id']}\n\
                    Post URL: {r['post_url']}\n\
                    Post Category: {category}.\n\
                    Score: {r['post_score']}\n"
                output.append(p)
            return "\n".join(output)
        else:
            return f"Searching r/{subreddit} did not find any posts:"


class RedditPostInput(BaseModel):
    """Input for the Reddit post reader tool."""
    post_id: str = Field(..., description="The ID of the Reddit post to read")

class RedditTool(BaseTool):
    """Tool for reading Reddit posts and comments."""
    name: str = "reddit_post_reader"
    description: str = "Read a Reddit post and its comments given the post ID"
    args_schema: Type[BaseModel] = RedditPostInput

    _reddit: praw.Reddit = PrivateAttr()

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        user_agent: str = "langchain_reddit_reader",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the Reddit tool with PRAW credentials.

        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent for Reddit API
            username: Reddit username (optional for read-only access)
            password: Reddit password (optional for read-only access)
        """
        super().__init__()
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            username=username,
            password=password,
        )

    def _parse_comment(self, comment, depth=0) -> Dict[str, Any]:
        """Parse a comment into a dictionary with relevant information."""
        return {
            "id": comment.id,
            "author": str(comment.author) if comment.author else "[deleted]",
            "body": comment.body,
            "score": comment.score,
            "created_utc": comment.created_utc,
            "depth": depth,
            "permalink": f"https://www.reddit.com{comment.permalink}",
        }

    def _run(self, post_id: str) -> Dict[str, Any]:
        """Run the tool to fetch a Reddit post and its comments."""
        try:
            # Fetch the submission directly using the post ID
            submission = self._reddit.submission(id=post_id)

            # Get post data
            post_data = {
                "id": submission.id,
                "title": submission.title,
                "author": str(submission.author) if submission.author else "[deleted]",
                "url": submission.url,
                "selftext": submission.selftext,
                "score": submission.score,
                "upvote_ratio": submission.upvote_ratio,
                "created_utc": submission.created_utc,
                "num_comments": submission.num_comments,
                "subreddit": submission.subreddit.display_name,
                "permalink": f"https://www.reddit.com{submission.permalink}",
            }

            # Get comments
            submission.comment_sort = "top"
            submission.comments.replace_more(limit=0)  # Skip "load more comments" links

            comments = []
            for idx, comment in enumerate(submission.comments):
                comments.append(self._parse_comment(comment))

            return {
                "post": post_data,
                "comments": comments,
            }

        except Exception as e:
            return {"error": f"Error fetching Reddit post: {str(e)}"}


reddit_tool = RedditTool(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)

reddit_post_reader_tool = Tool(
    name="reddit_post",
    func=reddit_tool._run,
    description=reddit_tool.description
)

# name=reddit_search
reddit_search_tool = RedditSearchRun(
    api_wrapper=RedditSearchAPIWrapperCustom(
        reddit_client_id=client_id,
        reddit_client_secret=client_secret,
        reddit_user_agent=user_agent,
    ),    
)

# Adapted code from /docs/modules/agents/how_to/sharedmemory_for_tools

from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_vertexai import ChatVertexAI
import vertexai
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate, MessagesPlaceholder


PROJECT_ID = "<redacted>"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-pro"
vertexai.init(project = PROJECT_ID, location=LOCATION)

tools = [
    reddit_search_tool,
    reddit_post_reader_tool,
]

memory = ConversationBufferMemory(memory_key="chat_history")

prefix="""# Reddit Search and Summary Agent

You are a specialized agent designed to search for Reddit posts on specific topics and provide clear, comprehensive summaries of both posts and their comments. Your goal is to help users quickly understand Reddit discussions without having to read through everything themselves.

## Your Tools:
1. **reddit_search**: Use this tool to find relevant Reddit posts based on keywords or topics. This returns basic information about posts but doesn't include full comments.
2. **reddit_post**: Use this tool to retrieve the complete content of a specific post including all comments. You must provide a post ID to this tool.

## Your Three-Step Process:
1. **Find Relevant Subreddits**: First, use the **reddit_search** tool with the user's query to identify which subreddits have the most relevant discussions on the topic
2. **Focus on Best Subreddit**: Once you've identified the most appropriate subreddit(s), perform a more targeted search within just that specific subreddit
3. **Analyze Top Posts**: Use the **reddit_post** tool to get complete information on the most relevant posts from that subreddit

## How You Should Respond:
1. First, clarify the user's search intent if needed
2. Use the **reddit_search** tool to find which subreddits have relevant content
3. Identify the most appropriate subreddit for the query and explain your choice briefly
4. Perform a focused search within that subreddit using the **reddit_search** tool
5. Select the most relevant posts and retrieve their full details using the **reddit_post** tool
6. Provide a summary that includes:
   - Name of the subreddit you focused on and why
   - Brief overview of the main post(s)
   - Key points from the discussion
   - Popular opinions and notable insights
   - Any consensus or disagreements
   - Unique or particularly helpful information
   - Urls of the final top posts which you used to arrive at your answer
7. Format the summary in a readable way with clear sections

## Guidelines for Summarization:
- Maintain neutrality - represent different viewpoints fairly
- Focus on substance over noise
- Preserve nuance from the original discussions
- Note the relative popularity of different opinions when relevant
- Indicate the recency of the posts when important for context
- Mention if a post has been edited or updated significantly"""

# Define a suffix for the agent prompt
suffix = """Remember to always follow the three-step process:
1. First find relevant subreddits
2. Then focus on the best subreddit
3. Finally analyze top posts from that subreddit

Begin your final summary with the subreddit you chose and why it was most appropriate for this query.

{chat_history}
Question: {input}
{agent_scratchpad}"""


prompt = StructuredChatAgent.create_prompt(
    prefix=prefix,
    tools=tools,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


llm = ChatVertexAI(model_name="gemini-2.0-flash", temperature=0.5, max_output_tokens=8192, verbose=True)
llm_chain = LLMChain(llm=llm, prompt=prompt)
agent = StructuredChatAgent(llm_chain=llm_chain, verbose=True, tools=tools)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, verbose=True, memory=memory, tools=tools, max_iterations=10
)

"""Sample Input
response = agent_chain.run(input="Must try street food in Ahmedabad.")
print(f"Ans: {response}")
"""

"""Sample output
> Finished chain.
Ans: Okay, here's a summary of must-try street foods in Ahmedabad based on Reddit discussions:

**Subreddit:** r/ahmedabad - This subreddit is the most relevant as it's dedicated to discussions about the city itself.

**General Observations:**
*   Ahmedabad is known for its vibrant street food scene.
*   Most street food is vegetarian.

**Must-Try Street Foods:**

1.  **Maskabun:**

    *   **Description:** A bun slathered with butter, often with fruit jam. It's a popular breakfast item.
    *   **Recommendations & Locations:**
        *   Generally available across Ahmedabad.
        *   One Redditor mentioned a spot near Khokhra Circle (possibly RK) as a famous place.
        *   Another user recommends Ma Rajwadi near Shivranjani cross roads for real butter maskabun.
        *   Another user recommends the maskabun near Zydus. Look for the orange umbrella from 6PM to 6AM.
    *   **Points of Discussion:**
        *   Some prefer it plain with butter, while others enjoy it with jam.
        *   Some consider the addition of cheese or chocolate to be sacrilegious.
        *   Some users dislike the masala version.
        *   Some users feel that the quality has declined in some places, with some vendors using margarine instead of real butter.
2.  **Sev Puri:**

    *   **Description:** A type of Indian street food snack. It is a puri which is loaded with diced potatoes, onions, various types of chutneys, and topped with sev.
    *   **Recommendations & Locations:**
        *   A popular recommendation is the sev puri wala near Domino's, Law Garden. Users praise the crunchiness of the sev.
    *   **Points of Discussion:**
        *   One user notes that the pakodi at this location isn't as good, but the sev puri is excellent.
3.  **Ghughra:**

    *   **Description:** A fried dumpling, usually filled with a sweet or savory mixture.
    *   **Recommendations & Locations:**
        *   Kapil na ghughra in Chandlodiya is mentioned, although one user laments the price increases over time.
4.  **Other Recommendations:**

    *   **Chhole Kulche:** Bajrang Chat wala near Panchvati Circle.
    *   **Pani Puri:** A pani puri wallah behind the High Court in Sola (only aloo no ragda).
    *   **Havmor Chanapuri:** Despite some users coming from places known for chana, they still enjoy Havmor's version.
    *   **Kamal's Pizza:** Near Kankaria.
    *   **Chilli Chips Chinese:** A street vendor near Vastrapur Lake.
    *   **Shivam Chinese & Novelty ka Pav Bhaji:** Vastrapur fountain char rasta.
    *   **Tea:** Iskon Gathiya ni baju vadi shop, VS ni gali ma Rajwadi and ST. Raipur ma Shri Ram ni fulvadi, English talkies na gota panchkuva.

**Additional Tips:**

*   Explore Manek Chowk (though some recent reviews are mixed).
*   Be aware that some areas may have restrictions on non-veg street food carts.
*   Some users recommend Delite Bakery near NID for egg puffs and other items.

**Posts Used:**

*   [https://www.reddit.com/r/ahmedabad/comments/1d2lzji/maskabun\_is\_under\_appreciated\_street\_food/](https://www.reddit.com/r/ahmedabad/comments/1d2lzji/maskabun_is_under_appreciated_street_food/)
*   [https://www.reddit.com/r/ahmedabad/comments/1fmmq29/underrated_street_food_places_which_you_keep/](https://www.reddit.com/r/ahmedabad/comments/1fmmq29/underrated_street_food_places_which_you_keep/)

"""