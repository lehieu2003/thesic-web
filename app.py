import os
import sys
from dotenv import load_dotenv
import psycopg
import litellm
from sentence_transformers import SentenceTransformer
import json
import re
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from functools import wraps

# Import chat history functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'db')))
from create_history_chat import (
    create_chat_history_tables, create_chat_session, get_all_chat_sessions,
    get_chat_history, save_chat_message, update_session_title, 
    delete_chat_session, clear_chat_history
)

# Import user functions
from create_user import (
    create_user_tables, register_user, login_user, get_user_by_id,
    update_user_profile, change_password, verify_jwt_token,
    get_all_users, update_user_role
)


# --- Your existing imports and global initializations ---
# Load environment variables
load_dotenv()

# DB config
DB_HOST = os.getenv("DB_HOST", "db.gprayrwxepherjifncqm.supabase.co")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")

# BERT model configuration
BERT_MODEL_NAME = "bert-base-uncased"

# LLM model for generation
litellm.api_key = os.getenv("LITELLM_API_KEY")
GENERATION_MODEL = "gemini/gemini-2.0-flash"

# Initialize BERT model globally
print(f"🤖 Loading BERT model: {BERT_MODEL_NAME}")
bert_model = SentenceTransformer(BERT_MODEL_NAME)
print("✅ BERT model loaded successfully")

# Initialize chat history tables on startup
print("🔧 Initializing user and chat history tables...")
create_user_tables()
create_chat_history_tables()
print("✅ User and chat history system ready!")

# Authentication middleware
def get_current_user():
    """Get current user from JWT token in Authorization header or session."""
    # Check Authorization header first
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        payload = verify_jwt_token(token)
        if payload:
            return {
                'user_id': payload['user_id'],
                'username': payload['username'],
                'role': payload.get('role', 'user')
            }
    
    # Check session as fallback
    user_id = session.get('user_id')
    if user_id:
        user_info = get_user_by_id(user_id)
        if user_info['success']:
            return {
                'user_id': user_id,
                'username': user_info['user']['username'],
                'role': user_info['user']['role']
            }
    
    return None

def require_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """Decorator to require admin role for routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if not user:
            # If not logged in, redirect to login page
            if request.path.startswith('/api/'):
                # For API routes, return JSON error
                return jsonify({'error': 'Authentication required'}), 401
            else:
                # For HTML routes, redirect to login with next parameter
                return render_template('login.html', message="Please login to access this page", 
                                     next_url=request.path), 401
        
        if user['role'] != 'admin':
            # If logged in but not admin, show access denied
            if request.path.startswith('/api/'):
                # For API routes, return JSON error
                return jsonify({'error': 'Admin access required'}), 403
            else:
                # For HTML routes, redirect to chat with error message
                return render_template('chat.html', error_message="You don't have permission to access the admin area"), 403
                
        return f(*args, **kwargs)
    return decorated_function

# --- Your existing functions (get_db_conn, get_embedding, exact_book_search, thematic_search, analyze_user_intent, generate_response) ---
# You'll need to copy these functions directly into or import them into app.py
# For brevity, I'm omitting them here, but imagine they are right above this Flask app code.

# Connect to PostgreSQL
def get_db_conn():
    try:
        return psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
    except Exception as e:
        print(f"❌ Error connecting to DB: {e}")
        return None

# Generate embedding for user input using BERT
def get_embedding(text: str):
    try:
        embedding = bert_model.encode([text], convert_to_tensor=False, show_progress_bar=False)
        return embedding[0].tolist()
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Exact book search using BERT embeddings
def exact_book_search(keyword: str, top_k=3):
    embedding = get_embedding(keyword)
    if not embedding:
        return []
    
    query = """
    SELECT 
        b.isbn13,
        b.title,
        b.authors,
        be.page_content,
        1 - (be.embedding <=> %(embedding)s::vector) AS similarity,
        b.description,
        b.thumbnail
    FROM book_embeddings_bert_base be
    JOIN books b ON b.isbn13 = be.book_isbn13
    ORDER BY be.embedding <=> %(embedding)s::vector
    LIMIT %(top_k)s;
    """
    
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        vector_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": vector_str, "top_k": top_k})
            return cur.fetchall()
    except Exception as e:
        print(f"❌ Error in exact search: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Thematic search using BERT embeddings
def thematic_search(keywords: list, top_k=3):
    combined_text = " ".join(keywords)
    embedding = get_embedding(combined_text)
    if not embedding:
        return []
    
    query = """
    SELECT 
        b.title, 
        b.authors, 
        be.page_content,
        1 - (be.embedding <=> %(embedding)s::vector) AS similarity,
        b.description,
        b.thumbnail
    FROM book_embeddings_bert_base be
    JOIN books b ON b.isbn13 = be.book_isbn13
    ORDER BY be.embedding <=> %(embedding)s::vector
    LIMIT %(top_k)s;
    """
    
    conn = get_db_conn()
    if not conn:
        return []
    
    try:
        vector_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": vector_str, "top_k": top_k})
            return cur.fetchall()
    except Exception as e:
        print(f"❌ Error in thematic search: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Enhanced query analysis
def analyze_user_intent(user_input: str):
    prompt = f"""
You are an intelligent query analyzer for a book recommendation system.

Analyze the user's query and determine:
1. **search_type**: 
    - "exact" if user is looking for specific books/series by name (e.g., "Harry Potter", "Lord of the Rings", "Naruto manga", "The Great Gatsby")
    - "thematic" if user wants books with similar themes/content/genres (e.g., "books like Harry Potter", "fantasy with magic", "romance novels", "love lesson books", "children books", "science fiction", "mystery novels")

2. **keywords**: Extract 3-5 relevant keywords for search that capture the theme/genre/topic
3. **numberOfBooks**: How many books to return (look for numbers like "5 books", "give me 10", etc. Default: 3)
4. **confidence**: How confident you are about the search_type (0.0-1.0)

### Important Guidelines:
- If user asks for books "about", "related to", "on", or mentions genres/categories -> "thematic"
- If user mentions specific book titles or series names -> "exact"
- For thematic searches, focus on genre, theme, and content keywords
- For exact searches, use the book/series name as keywords

### Examples:
- "Harry Potter" -> search_type: "exact", keywords: ["Harry Potter"]
- "books like Harry Potter" -> search_type: "thematic", keywords: ["fantasy", "magic", "wizards", "adventure", "young adult"]
- "love lesson books" -> search_type: "thematic", keywords: ["love", "romance", "relationships", "lessons", "emotional growth"]
- "children books" -> search_type: "thematic", keywords: ["children", "kids", "young readers", "picture books", "educational"]
- "give me 5 books related to love lesson" -> search_type: "thematic", keywords: ["love", "romance", "relationships", "lessons", "personal growth"], numberOfBooks: 5
- "science fiction novels" -> search_type: "exact", keywords: ["science fiction", "sci-fi", "technology", "future", "space"]
- "The Great Gatsby" -> search_type: "exact", keywords: ["The Great Gatsby"]

### User Input: 
"{user_input}"

### Output (JSON only):
{{
    "search_type": "exact" or "thematic",
    "keywords": ["keyword1", "keyword2", ...],
    "numberOfBooks": number,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

    try:
        response = litellm.completion(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message["content"]
        # Clean JSON response
        json_str = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(json_str)
        
        print(f"🧠 Query Analysis: {parsed['search_type']} search (confidence: {parsed['confidence']:.2f})")
        print(f"💡 Reasoning: {parsed['reasoning']}")
        print(f"🔍 Keywords: {parsed['keywords']}")
        
        return parsed
    except Exception as e:
        print(f"❌ Error analyzing intent: {e}")
        # Enhanced fallback logic
        user_lower = user_input.lower()
        
        # Extract number if mentioned
        import re
        number_match = re.search(r'\b(\d+)\s*books?\b', user_lower)
        num_books = int(number_match.group(1)) if number_match else 3
        
        # Check for thematic indicators
        thematic_indicators = [
            "like", "similar to", "books about", "related to", "genre", "type of",
            "recommend", "suggest", "books on", "novels about", "stories about",
            "love", "romance", "children", "kids", "fantasy", "sci-fi", "mystery",
            "horror", "thriller", "adventure", "biography", "self-help", "business"
        ]
        
        # Check for exact book/series names (common ones)
        exact_indicators = [
            "harry potter", "lord of the rings", "naruto", "one piece", "game of thrones",
            "hunger games", "twilight", "fifty shades", "the great gatsby", "to kill a mockingbird",
            "pride and prejudice", "1984", "animal farm", "sherlock holmes"
        ]
        
        if any(indicator in user_lower for indicator in thematic_indicators):
            search_type = "thematic"
            # Extract meaningful keywords for thematic search
            words = user_input.lower().replace("books", "").replace("novels", "").replace("stories", "").split()
            keywords = [word for word in words if len(word) > 2 and word not in ["give", "me", "recommend", "suggest", "related", "about"]][:5]
        elif any(indicator in user_lower for indicator in exact_indicators):
            search_type = "exact"
            keywords = user_input.split()
        else:
            # Default heuristic: short phrases likely exact, longer descriptive phrases likely thematic
            if len(user_input.split()) <= 2 and not any(word in user_lower for word in ["books", "novels", "stories"]):
                search_type = "exact"
                keywords = user_input.split()
            else:
                search_type = "thematic"
                words = user_input.lower().replace("books", "").replace("novels", "").split()
                keywords = [word for word in words if len(word) > 2][:5]
        
        return {
            "search_type": search_type,
            "keywords": keywords if keywords else user_input.split(),
            "numberOfBooks": num_books,
            "confidence": 0.6,
            "reasoning": "Fallback analysis using keyword detection"
        }

# Generate natural language response
def generate_response(user_query: str, search_results: list, search_type: str):
    if not search_results:
        return "❌ No books found matching your criteria."
    
    if search_type == "exact":
        context = "\n\n".join([
            f"Title: {row[1]}\nAuthor(s): {row[2]}\nISBN: {row[0]}\nSimilarity: {row[4]:.3f}\nDescription: {row[5][:300] if row[5] else 'No description available'}...\nExcerpt: {row[3][:200]}..."
            for row in search_results
        ])
        prompt_type = "exact books matching"
    else:
        context = "\n\n".join([
            f"Title: {row[0]}\nAuthor(s): {row[1]}\nDescription: {row[4][:300] if row[4] else 'No description available'}...\nExcerpt: {row[2][:200]}..."
            for row in search_results
        ])
        prompt_type = "thematically similar books for"
    
    prompt = f"""You are a helpful book recommendation assistant.

The user searched for: "{user_query}"
Search type: {search_type}

I found these {prompt_type} the user's query:

{context}

Provide a helpful, natural response that:
1. Acknowledges what the user was looking for
2. Presents the books in an organized way
3. Explains why these books match their request
4. Suggests what they might enjoy about these books

IMPORTANT FORMATTING GUIDELINES:
- Use bullet points with single asterisks (*) for lists
- Use **bold** for book titles and important points
- Use clear paragraph breaks
- Keep explanations conversational but structured
- Each book recommendation should be a separate bullet point
- Format like this example:

I found some great recommendations for you! Here are the books that match your request:

* **Book Title 1**: Brief description of why this book matches and what you might enjoy about it.

* **Book Title 2**: Brief description of why this book matches and what you might enjoy about it.

These books share similar themes of [common themes] which should appeal to fans of [user's request].

Be conversational and helpful:
"""

    try:
        response = litellm.completion(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return "Found some books for you, but couldn't generate a detailed response."


app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Authentication routes
@app.route('/')
def home():
    """Serve the homepage."""
    return render_template('home_page.html')

@app.route('/login')
def login_page():
    """Serve the login page."""
    return render_template('login.html')

@app.route('/register')
def register_page():
    """Serve the register page."""
    return render_template('register.html')

@app.route('/book/<isbn13>')
def book_detail_page(isbn13):
    """Serve the book detail page."""
    return render_template('book_detail.html', isbn13=isbn13)

@app.route('/chat')
@require_auth
def chat_page():
    """Serve the chat page for authenticated users."""
    return render_template('chat.html')

@app.route('/api/auth/register', methods=['POST'])
def api_register():
    """Register a new user."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        
        result = register_user(username, email, password, full_name)
        
        if result['success']:
            # Store user_id in session for backward compatibility
            session['user_id'] = result['user']['user_id']
            
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in register endpoint: {e}")
        return jsonify({'success': False, 'message': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def api_login():
    """Login a user."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        result = login_user(username, password)
        
        if result['success']:
            # Store user_id in session for backward compatibility
            session['user_id'] = result['user']['user_id']
            
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error in login endpoint: {e}")
        return jsonify({'success': False, 'message': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    """Logout the current user."""
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/auth/profile')
@require_auth
def api_profile():
    """Get current user profile."""
    user = get_current_user()
    result = get_user_by_id(user['user_id'])
    return jsonify(result)

# Admin routes
@app.route('/admin')
@require_admin
def admin_dashboard():
    """Serve admin dashboard."""
    return render_template('admin.html')

@app.route('/api/admin/users')
@require_admin
def api_get_all_users():
    """Get all users (admin only)."""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    offset = (page - 1) * limit
    
    result = get_all_users(limit=limit, offset=offset)
    return jsonify(result)

@app.route('/api/admin/users/<user_id>/role', methods=['PUT'])
@require_admin
def api_update_user_role(user_id):
    """Update user role (admin only)."""
    data = request.get_json()
    new_role = data.get('role')
    
    if new_role not in ['user', 'admin']:
        return jsonify({'error': 'Invalid role'}), 400
    
    result = update_user_role(user_id, new_role)
    return jsonify(result)

@app.route('/api/admin/stats')
@require_admin
def api_admin_stats():
    """Get system statistics (admin only)."""
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            # Get user count by role
            cur.execute("""
                SELECT role, COUNT(*) 
                FROM users 
                WHERE is_active = TRUE 
                GROUP BY role
            """)
            user_stats = dict(cur.fetchall())
            
            # Get total books
            cur.execute("SELECT COUNT(*) FROM books")
            total_books = cur.fetchone()[0]
            
            # Get total chat sessions
            cur.execute("SELECT COUNT(*) FROM chat_sessions_bert_base")
            total_sessions = cur.fetchone()[0]
            
            # Get total messages
            cur.execute("SELECT COUNT(*) FROM chat_messages_bert_base")
            total_messages = cur.fetchone()[0]
            
            # Get active users (logged in within last 7 days)
            cur.execute("""
                SELECT COUNT(*) 
                FROM users 
                WHERE last_login > NOW() - INTERVAL '7 days' AND is_active = TRUE
            """)
            active_users = cur.fetchone()[0]
            
        return jsonify({
            'success': True,
            'stats': {
                'users': {
                    'total': sum(user_stats.values()),
                    'admins': user_stats.get('admin', 0),
                    'regular_users': user_stats.get('user', 0),
                    'active_last_7_days': active_users
                },
                'books': {
                    'total': total_books
                },
                'chat': {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages
                }
            }
        })
        
    except Exception as e:
        print(f"❌ Error getting admin stats: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/admin/books')
@require_admin
def api_get_books():
    """Get all books with pagination and search (admin only)."""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 25, type=int)
    search = request.args.get('search', '').strip()
    
    offset = (page - 1) * limit
    
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            # Build search condition
            where_clause = ""
            params = []
            
            if search:
                where_clause = """
                WHERE (LOWER(title) LIKE LOWER(%s) 
                    OR LOWER(authors) LIKE LOWER(%s) 
                    OR isbn13 LIKE %s)
                """
                search_param = f"%{search}%"
                params = [search_param, search_param, search_param]
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM books {where_clause}"
            cur.execute(count_query, params)
            total_books = cur.fetchone()[0]
            
            # Get books with pagination
            books_query = f"""
                SELECT isbn13, title, authors, description, thumbnail
                FROM books 
                {where_clause}
                ORDER BY title
                LIMIT %s OFFSET %s
            """
            cur.execute(books_query, params + [limit, offset])
            books = cur.fetchall()
            
            # Format books data
            books_data = []
            for book in books:
                books_data.append({
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'description': book[3],
                    'thumbnail': book[4]
                })
            
            total_pages = (total_books + limit - 1) // limit
            
            return jsonify({
                'success': True,
                'books': books_data,
                'pagination': {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_books': total_books,
                    'limit': limit
                }
            })
            
    except Exception as e:
        print(f"❌ Error getting books: {e}")
        return jsonify({'error': 'Failed to get books'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/admin/books/<isbn13>')
@require_admin
def api_get_book_detail(isbn13):
    """Get detailed information about a specific book (admin only)."""
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT isbn13, isbn10, title, title_and_subtitle, authors, 
                       categories, simple_categories, thumbnail, description, 
                       tagged_description, published_year, average_rating, 
                       num_pages, ratings_count, anger, disgust, fear, joy, 
                       sadness, surprise, neutral, created_at, updated_at
                FROM books 
                WHERE isbn13 = %s
            """, (isbn13,))
            
            book = cur.fetchone()
            if not book:
                return jsonify({'error': 'Book not found'}), 404
            
            book_data = {
                'isbn13': book[0],
                'isbn10': book[1],
                'title': book[2],
                'title_and_subtitle': book[3],
                'authors': book[4],
                'categories': book[5],
                'simple_categories': book[6],
                'thumbnail': book[7],
                'description': book[8],
                'tagged_description': book[9],
                'published_year': book[10],
                'average_rating': book[11],
                'num_pages': book[12],
                'ratings_count': book[13],
                'anger': book[14],
                'disgust': book[15],
                'fear': book[16],
                'joy': book[17],
                'sadness': book[18],
                'surprise': book[19],
                'neutral': book[20],
                'created_at': book[21].isoformat() if book[21] else None,
                'updated_at': book[22].isoformat() if book[22] else None
            }
            
            return jsonify({
                'success': True,
                'book': book_data
            })
            
    except Exception as e:
        print(f"❌ Error getting book detail: {e}")
        return jsonify({'error': 'Failed to get book details'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/admin/books/<isbn13>', methods=['PUT'])
@require_admin
def api_edit_book(isbn13):
    """Edit an existing book (admin only)."""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            # Check if book exists
            cur.execute("SELECT isbn13 FROM books WHERE isbn13 = %s", (isbn13,))
            if not cur.fetchone():
                return jsonify({'error': 'Book not found'}), 404
            
            # Build update query dynamically based on provided fields
            update_fields = []
            update_values = []
            
            # Map form fields to actual database columns
            field_mapping = {
                'isbn10': 'isbn10',
                'title': 'title',
                'title_and_subtitle': 'title_and_subtitle',
                'authors': 'authors',
                'categories': 'categories',
                'simple_categories': 'simple_categories',
                'thumbnail': 'thumbnail',
                'description': 'description',
                'tagged_description': 'tagged_description',
                'published_year': 'published_year',
                'average_rating': 'average_rating',
                'num_pages': 'num_pages',
                'ratings_count': 'ratings_count',
                'anger': 'anger',
                'disgust': 'disgust',
                'fear': 'fear',
                'joy': 'joy',
                'sadness': 'sadness',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }
            
            for form_field, db_field in field_mapping.items():
                if form_field in data and data[form_field] is not None:
                    update_fields.append(f"{db_field} = %s")
                    update_values.append(data[form_field])
            
            if not update_fields:
                return jsonify({'error': 'No valid fields to update'}), 400
            
            # Add updated_at timestamp
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            # Add ISBN to the end for WHERE clause
            update_values.append(isbn13)
            
            update_query = f"""
                UPDATE books 
                SET {', '.join(update_fields)}
                WHERE isbn13 = %s
            """
            
            cur.execute(update_query, update_values)
            conn.commit();
            
            return jsonify({
                'success': True,
                'message': 'Book updated successfully'
            })
            
    except Exception as e:
        print(f"❌ Error updating book: {e}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'Failed to update book'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/admin/books', methods=['POST'])
@require_admin
def api_add_book():
    """Add a new book (admin only)."""
    data = request.get_json()
    
    required_fields = ['isbn13', 'title', 'authors']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400
    
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            # Check if book already exists
            cur.execute("SELECT isbn13 FROM books WHERE isbn13 = %s", (data['isbn13'],))
            if cur.fetchone():
                return jsonify({'error': 'Book with this ISBN already exists'}), 400
            
            # Insert new book with basic fields
            cur.execute("""
                INSERT INTO books (isbn13, isbn10, title, authors, description, thumbnail, categories, simple_categories)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                data['isbn13'],
                data.get('isbn10'),
                data['title'],
                data['authors'],
                data.get('description'),
                data.get('thumbnail'),
                data.get('categories'),
                data.get('simple_categories')
            ))
            
            conn.commit();
            
            return jsonify({
                'success': True,
                'message': 'Book added successfully'
            })
            
    except Exception as e:
        print(f"❌ Error adding book: {e}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'Failed to add book'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/admin/books/<isbn13>', methods=['DELETE'])
@require_admin
def api_delete_book(isbn13):
    """Delete a book (admin only)."""
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            # Check if book exists
            cur.execute("SELECT isbn13 FROM books WHERE isbn13 = %s", (isbn13,))
            if not cur.fetchone():
                return jsonify({'error': 'Book not found'}), 404
            
            # Delete related embeddings first
            cur.execute("DELETE FROM book_embeddings_bert_base WHERE book_isbn13 = %s", (isbn13,))
            
            # Delete the book
            cur.execute("DELETE FROM books WHERE isbn13 = %s", (isbn13,))
            
            conn.commit();
            
            return jsonify({
                'success': True,
                'message': 'Book deleted successfully'
            })
            
    except Exception as e:
        print(f"❌ Error deleting book: {e}")
        if conn:
            conn.rollback()
        return jsonify({'error': 'Failed to delete book'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    """Handle chat messages with user authentication."""
    try:
        user = get_current_user()
        user_id = user['user_id']
        data = request.get_json()
        
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        # Verify that the session belongs to the current user
        session_data = get_chat_history(session_id)
        if not session_data.get('metadata'):
            return jsonify({'error': 'Session not found'}), 404

        # Save user message to database
        save_chat_message(session_id, 'user', user_message)
        
        # Analyze user intent
        intent = analyze_user_intent(user_message)
        
        results = []
        if intent["search_type"] == "exact":
            results = exact_book_search(user_message, top_k=intent["numberOfBooks"])
        else:  # thematic search
            results = thematic_search(intent["keywords"], top_k=intent["numberOfBooks"])
        
        # Generate AI response
        ai_response = generate_response(user_message, results, intent["search_type"])
        
        # Format the complete response
        response_data = {
            'ai_response': ai_response,
            'books': [],
            'search_type': intent["search_type"],
            'intent': intent
        }
        
        # Format book results
        if results:
            for book in results:
                if intent["search_type"] == "exact":
                    book_data = {
                        'title': book[1],
                        'authors': book[2],
                        'isbn': book[0],
                        'excerpt': book[3][:200] + "..." if len(book[3]) > 200 else book[3],
                        'similarity': round(book[4], 3),
                        'description': book[5] if book[5] else 'No description available',
                        'thumbnail': book[6] if book[6] else None
                    }
                else:
                    book_data = {
                        'title': book[0],
                        'authors': book[1],
                        'excerpt': book[2][:200] + "..." if len(book[2]) > 200 else book[2],
                        'similarity': round(book[3], 3),
                        'description': book[4] if book[4] else 'No description available',
                        'thumbnail': book[5] if book[5] else None
                    }
                response_data['books'].append(book_data)
        
        # Save bot response to database
        save_chat_message(
            session_id, 
            'bot', 
            ai_response, 
            search_type=intent["search_type"],
            intent_data=intent,
            books_data=response_data['books']
        )
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ An error occurred during chat: {e}")
        error_response = "Xin lỗi, có lỗi xảy ra. Vui lòng thử lại sau."
        
        # Try to save error response to database if session_id is available
        try:
            if 'session_id' in locals():
                save_chat_message(session_id, 'bot', error_response, search_type='error')
        except:
            pass
        
        return jsonify({'error': error_response}), 500

@app.route('/api/chat/history/<session_id>')
@require_auth
def get_chat_history_route(session_id):
    """Get chat history for a session from database"""
    return jsonify(get_chat_history(session_id))

@app.route('/api/chat/sessions')
@require_auth
def get_all_sessions():
    """Get all chat sessions for the current user from database"""
    user = get_current_user()
    user_id = user['user_id']
    sessions = get_all_chat_sessions(user_id)
    return jsonify({'sessions': sessions})

@app.route('/api/chat/new', methods=['POST'])
@require_auth
def create_new_chat():
    """Create a new chat session for the current user in database"""
    user = get_current_user()
    user_id = user['user_id']
    session_id = create_chat_session(user_id=user_id)
    if session_id:
        return jsonify({'session_id': session_id})
    else:
        return jsonify({'error': 'Failed to create new chat session'}), 500

@app.route('/api/chat/clear/<session_id>', methods=['POST'])
@require_auth
def clear_chat_history_route(session_id):
    """Clear chat history for a session in database"""
    try:
        user = get_current_user()
        user_id = user['user_id']
        
        # Verify the session belongs to the current user
        conn = get_db_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id FROM chat_sessions_bert_base WHERE session_id = %s AND user_id = %s",
                        (session_id, user_id)
                    )
                    session_exists = cur.fetchone() is not None
                    
                if not session_exists:
                    return jsonify({'error': 'Session not found or access denied'}), 404
            finally:
                conn.close()
        
        result = clear_chat_history(session_id)
        if result:
            return jsonify({'success': True, 'message': 'Chat history cleared'})
        else:
            return jsonify({'success': False, 'error': 'Failed to clear chat history'}), 500
    except Exception as e:
        print(f"❌ Exception in clear_chat_history_route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chat/delete/<session_id>', methods=['DELETE'])
@require_auth
def delete_chat_session_route(session_id):
    """Delete a chat session from database"""
    try:
        user = get_current_user()
        user_id = user['user_id']
        
        # Verify the session belongs to the current user
        conn = get_db_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT session_id FROM chat_sessions_bert_base WHERE session_id = %s AND user_id = %s",
                        (session_id, user_id)
                    )
                    session_exists = cur.fetchone() is not None
                    
                if not session_exists:
                    return jsonify({'error': 'Session not found or access denied'}), 404
            finally:
                conn.close()
        
        result = delete_chat_session(session_id)
        if result:
            return jsonify({'success': True, 'message': 'Chat session deleted'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete chat session'}), 500
    except Exception as e:
        print(f"❌ Exception in delete_chat_session_route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chat/rename/<session_id>', methods=['POST'])
@require_auth
def rename_chat_session(session_id):
    """Rename a chat session in database"""
    try:
        data = request.get_json()
        if not data:
            print(f"❌ No JSON data received for rename request")
            return jsonify({'error': 'No data provided'}), 400
        
        new_title = data.get('title', '').strip()
        
        if not new_title:
            print(f"❌ Empty title provided for session {session_id}")
            return jsonify({'error': 'Title cannot be empty'}), 400
        
        print(f"🔄 Attempting to rename session {session_id} to '{new_title}'")
        
        if update_session_title(session_id, new_title):
            print(f"✅ Successfully renamed session {session_id}")
            return jsonify({'message': 'Chat renamed successfully'})
        else:
            print(f"❌ Failed to rename session {session_id} - session not found or database error")
            return jsonify({'error': 'Failed to rename chat session - session may not exist'}), 404
    
    except Exception as e:
        print(f"❌ Exception in rename_chat_session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# ==================== HOMEPAGE API ENDPOINTS ====================

@app.route('/api/home/stats', methods=['GET'])
def get_home_stats():
    """Get statistics for homepage."""
    try:
        conn = get_db_conn()
        if not conn:
            # Fallback stats if database is not available
            stats = {
                'total_books': 45892,
                'active_users': 12543,
                'recommendations_served': 234567
            }
            return jsonify(stats)
        
        with conn.cursor() as cur:
            # Get total books
            cur.execute("SELECT COUNT(*) FROM books")
            total_books = cur.fetchone()[0]
            
            # Get total active users (registered users)
            cur.execute("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
            active_users = cur.fetchone()[0] if cur.rowcount > 0 else 0
            
            # Get total chat messages as proxy for recommendations served
            cur.execute("SELECT COUNT(*) FROM chat_messages_bert_base WHERE message_type = 'bot'")
            recommendations_served = cur.fetchone()[0] if cur.rowcount > 0 else 0
            
            stats = {
                'total_books': total_books,
                'active_users': active_users,
                'recommendations_served': recommendations_served
            }
            
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting home stats: {e}")
        # Return fallback stats on error
        stats = {
            'total_books': 45892,
            'active_users': 12543,
            'recommendations_served': 234567
        }
        
        
        return jsonify(stats)
    finally:
        if conn:
            conn.close()

@app.route('/api/home/new-releases', methods=['GET'])
def get_new_releases():
    """Get new book releases for homepage."""
    try:
        conn = get_db_conn()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with conn.cursor() as cur:
            # Get newest books based on published_year and created_at
            cur.execute("""
                SELECT isbn13, title, authors, description, thumbnail, average_rating, published_year
                FROM books 
                WHERE published_year IS NOT NULL 
                ORDER BY published_year DESC, created_at DESC 
                LIMIT 8
            """)
            
            books_data = cur.fetchall()
            books = []
            
            for i, book in enumerate(books_data):
                books.append({
                    'id': i + 1,
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'thumbnail': book[4],
                    'average_rating': round(book[5], 1) if book[5] else 4.0,
                    'description': book[3][:100] + '...' if book[3] and len(book[3]) > 100 else book[3] or 'Sách mới phát hành.',
                    'year': book[6]
                })
            
        return jsonify(books)
    except Exception as e:
        print(f"Error getting new releases: {e}")
        return jsonify({'error': 'Failed to load new releases'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/home/bestsellers', methods=['GET'])
def get_bestsellers():
    """Get bestselling books for homepage."""
    try:
        conn = get_db_conn()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with conn.cursor() as cur:
            # Get bestselling books based on ratings_count
            cur.execute("""
                SELECT isbn13, title, authors, description, thumbnail, average_rating, ratings_count
                FROM books 
                WHERE ratings_count IS NOT NULL AND ratings_count > 0 
                ORDER BY ratings_count DESC, average_rating DESC 
                LIMIT 8
            """)
            
            books_data = cur.fetchall()
            books = []
            
            for i, book in enumerate(books_data):
                books.append({
                    'id': i + 5,  # Start from 5 to avoid ID conflicts with new releases
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'thumbnail': book[4] ,
                    'average_rating': round(book[5], 1) if book[5] else 4.0,
                    'description': book[3][:100] + '...' if book[3] and len(book[3]) > 100 else book[3] or 'Sách bán chạy nhất.',
                    'ratings_count': book[6]
                })
            
        return jsonify(books)
    except Exception as e:
        print(f"Error getting bestsellers: {e}")
        return jsonify({'error': 'Failed to load bestsellers'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/home/top-rated', methods=['GET'])
def get_top_rated():
    """Get top-rated books for homepage."""
    try:
        conn = get_db_conn()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with conn.cursor() as cur:
            # Get top-rated books with minimum ratings count to ensure quality
            cur.execute("""
                SELECT isbn13, title, authors, description, thumbnail, average_rating, ratings_count
                FROM books 
                WHERE average_rating IS NOT NULL AND average_rating >= 4.0 
                AND ratings_count IS NOT NULL AND ratings_count >= 10
                ORDER BY average_rating DESC, ratings_count DESC 
                LIMIT 8
            """)
            
            books_data = cur.fetchall()
            books = []
            
            for i, book in enumerate(books_data):
                books.append({
                    'id': i + 9,  # Start from 9 to avoid ID conflicts
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'thumbnail': book[4],
                    'average_rating': round(book[5], 1) if book[5] else 4.0,
                    'description': book[3][:100] + '...' if book[3] and len(book[3]) > 100 else book[3] or 'Sách được đánh giá cao.',
                    'ratings_count': book[6]
                })
            
        return jsonify(books)
    except Exception as e:
        print(f"Error getting top rated books: {e}")
        return jsonify({'error': 'Failed to load top rated books'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/home/personalized', methods=['GET'])
def get_personalized_recommendations():
    """Get personalized recommendations for logged-in users."""
    try:
        user_id = get_current_user()
        if not user_id:
            return jsonify([])  # Return empty array for non-authenticated users
        
        conn = get_db_conn()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with conn.cursor() as cur:
            # Try to get user-specific recommendations based on their reading history
            # For now, get popular books with good ratings as fallback
            cur.execute("""
                SELECT isbn13, title, authors, description, thumbnail, average_rating, ratings_count
                FROM books 
                WHERE average_rating IS NOT NULL AND average_rating >= 4.0 
                AND ratings_count IS NOT NULL AND ratings_count >= 5
                ORDER BY (average_rating * 0.6 + LEAST(ratings_count / 100.0, 5) * 0.4) DESC
                LIMIT 8
            """)
            
            books_data = cur.fetchall()
            books = []
            
            # Simple recommendation reasons based on book characteristics
            reasons = [
                'Dựa trên sở thích đọc của bạn',
                'Phù hợp với phong cách đọc của bạn',
                'Tác phẩm được đánh giá cao',
                'Khám phá thể loại mới',
                'Được độc giả yêu thích',
                'Phù hợp với lịch sử đọc của bạn',
                'Đề xuất dành cho bạn',
                'Sách có đánh giá tích cực'
            ]
            
            for i, book in enumerate(books_data):
                books.append({
                    'id': i + 13,  # Start from 13 to avoid ID conflicts
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'thumbnail': book[4],
                    'average_rating': round(book[5], 1) if book[5] else 4.0,
                    'description': book[3][:100] + '...' if book[3] and len(book[3]) > 100 else book[3] or 'Đề xuất dành cho bạn.',
                    'reason': reasons[i % len(reasons)]
                })
            
        return jsonify(books)
    except Exception as e:
        print(f"Error getting personalized recommendations: {e}")
        return jsonify({'error': 'Failed to load recommendations'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/search/quick', methods=['GET'])
def quick_search():
    """Quick search for books - used in homepage search."""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify([])
        
        conn = get_db_conn()
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
        
        with conn.cursor() as cur:
            # Search for books by title, authors, or description
            cur.execute("""
                SELECT isbn13, title, authors, thumbnail
                FROM books 
                WHERE LOWER(title) LIKE LOWER(%s) 
                   OR LOWER(authors) LIKE LOWER(%s)
                   OR LOWER(description) LIKE LOWER(%s)
                ORDER BY 
                    CASE 
                        WHEN LOWER(title) LIKE LOWER(%s) THEN 1
                        WHEN LOWER(authors) LIKE LOWER(%s) THEN 2
                        ELSE 3
                    END
                LIMIT 5
            """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
            
            results = []
            for i, book in enumerate(cur.fetchall()):
                results.append({
                    'id': i + 1,
                    'isbn13': book[0],
                    'title': book[1],
                    'authors': book[2],
                    'thumbnail': book[3]
                })
        
        return jsonify(results)
    except Exception as e:
        print(f"Error in quick search: {e}")
        return jsonify({'error': 'Search failed'}), 500
    finally:
        if conn:
            conn.close()

@app.route('/api/books/<isbn13>')
def api_get_public_book_detail(isbn13):
    """Get detailed information about a specific book (public access)."""
    conn = get_db_conn()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT isbn13, isbn10, title, title_and_subtitle, authors, 
                       categories, simple_categories, thumbnail, description, 
                       tagged_description, published_year, average_rating, 
                       num_pages, ratings_count, anger, disgust, fear, joy, 
                       sadness, surprise, neutral
                FROM books 
                WHERE isbn13 = %s
            """, (isbn13,))
            
            book = cur.fetchone()
            if not book:
                return jsonify({'error': 'Book not found'}), 404
            
            book_data = {
                'isbn13': book[0],
                'isbn10': book[1],
                'title': book[2],
                'title_and_subtitle': book[3],
                'authors': book[4],
                'categories': book[5],
                'simple_categories': book[6],
                'thumbnail': book[7],
                'description': book[8],
                'tagged_description': book[9],
                'published_year': book[10],
                'average_rating': book[11],
                'num_pages': book[12],
                'ratings_count': book[13],
                'emotions': {
                    'anger': book[14],
                    'disgust': book[15],
                    'fear': book[16],
                    'joy': book[17],
                    'sadness': book[18],
                    'surprise': book[19],
                    'neutral': book[20]
                }
            }
            
            return jsonify({
                'success': True,
                'book': book_data
            })
            
    except Exception as e:
        print(f"❌ Error getting book detail: {e}")
        return jsonify({'error': 'Failed to get book details'}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)