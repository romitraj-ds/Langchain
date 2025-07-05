import streamlit as st
from restaurant_agent import get_restaurant_name_and_items


st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a cuisine", ["Italian", "Chinese", "Mexican", "Indian", "French"])

if cuisine:
    response = get_restaurant_name_and_items(cuisine)

    st.header(f"🍽️ {response['restaurant_name']}")
    menu = response['menu']

    for category, items in menu.items():
        st.subheader(f"📋 {category}")

        with st.container():
            if isinstance(items, list) and len(items) > 0:
                if isinstance(items[0], dict) and "name" in items[0] and "description" in items[0]:
                    for item in items:
                        st.markdown(f"**{item['name']}**")
                        st.markdown(f"_{item['description']}_")
                        st.divider()
                elif isinstance(items[0], list) and len(items[0]) >= 2:
                    for item in items:
                        st.markdown(f"**{item[0]}**")
                        st.markdown(f"_{item[1]}_")
                        st.divider()
                else:
                    for item in items:
                        st.markdown(f"• {item}")
            else:
                st.write("No items in this category")

        st.write("")
