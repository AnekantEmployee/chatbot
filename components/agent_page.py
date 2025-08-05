import streamlit as st
import time
from backend.agent import compiled_graph, search_chain
from constants.index import AGENT_CONFIG_ID
from langchain_core.messages import HumanMessage


class StreamingUI:
    def __init__(self):
        self.query_container = None
        self.sources_container = None
        self.answer_container = None
        self.sources_data = []

    def setup_containers(self):
        """Setup streaming containers"""
        st.subheader("🔍 Live Search Progress")
        self.query_container = st.empty()
        self.sources_container = st.empty()
        self.answer_container = st.empty()

    def update_display(self, update_type, content):
        """Handle streaming updates"""
        if update_type == "query":
            self.query_container.info(content)

        elif update_type == "status":
            self.query_container.info(content)

        elif update_type == "source":
            # Extract title and URL for button
            if "(" in content and content.endswith(")"):
                title = content.split(" (")[0]
                url = content.split(" (")[1][:-1]
            else:
                title = content
                url = ""

            self.sources_data.append({"title": title, "url": url})

            # Display sources as buttons
            with self.sources_container.container():
                st.success("📚 **Sources Found:**")

                # Create columns for buttons (2 per row)
                cols = st.columns(2)
                for i, source in enumerate(self.sources_data[:4]):
                    col_idx = i % 2
                    with cols[col_idx]:
                        if source["url"]:
                            st.link_button(
                                (
                                    f"🔗 {source['title'][:40]}..."
                                    if len(source["title"]) > 40
                                    else f"🔗 {source['title']}"
                                ),
                                source["url"],
                                use_container_width=True,
                            )
                        else:
                            st.button(
                                (
                                    f"📄 {source['title'][:40]}..."
                                    if len(source["title"]) > 40
                                    else f"📄 {source['title']}"
                                ),
                                disabled=True,
                                use_container_width=True,
                            )

        elif update_type == "answer":
            with self.answer_container.container():
                st.success("✅ **Final Answer:**")
                st.write(content)

        elif update_type == "error":
            self.answer_container.error(f"❌ {content}")


def agent_page_component():
    st.title("🚀 Streaming Search Agent")
    st.markdown("Watch live updates as the agent searches and processes information!")

    # Initialize session state
    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    # Display chat history
    for message in st.session_state["agent_history"]:
        with st.chat_message(message["role"]):
            if message["role"] == "ai" and "search_data" in message:
                display_final_response(message["search_data"])
            else:
                st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        # Add user message
        st.session_state["agent_history"].append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("ai"):
            # Setup streaming UI
            streaming_ui = StreamingUI()
            streaming_ui.setup_containers()

            try:
                # Execute search chain with streaming
                search_data = search_chain.execute_search_chain(
                    user_input, update_callback=streaming_ui.update_display
                )

                # Store in history
                st.session_state["agent_history"].append(
                    {
                        "role": "ai",
                        "content": search_data["agent_answer"],
                        "search_data": search_data,
                    }
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state["agent_history"].append(
                    {
                        "role": "ai",
                        "content": f"Error: {str(e)}",
                        "search_data": {"error": str(e)},
                    }
                )


def display_final_response(search_data):
    """Display clean final response with source buttons"""
    if search_data.get("error"):
        st.error(f"❌ {search_data['error']}")

    # Main answer
    st.success("✅ **Final Answer:**")
    st.write(search_data.get("agent_answer", "No response"))

    # Sources as clean buttons
    results = search_data.get("search_results", [])
    if results:
        st.markdown("**📚 Sources:**")

        # Create columns for source buttons (2 per row)
        cols = st.columns(2)

        for i, result in enumerate(results[:5]):  # Ensure max 5
            col_idx = i % 2
            title = result.get("title", "No title")
            url = result.get("url", "")

            with cols[col_idx]:
                if url:
                    st.link_button(
                        f"🔗 {title[:40]}..." if len(title) > 40 else f"🔗 {title}",
                        url,
                        use_container_width=True,
                    )
                else:
                    st.button(
                        f"📄 {title[:40]}..." if len(title) > 40 else f"📄 {title}",
                        disabled=True,
                        use_container_width=True,
                    )
