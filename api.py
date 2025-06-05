# API endpoints for book recommendation pages
import os
import psycopg
import traceback
from dotenv import load_dotenv
from flask import request, jsonify
import math

# Load environment variables
load_dotenv()

# Database configuration
DB_HOST = os.getenv("DB_HOST", "db.gprayrwxepherjifncqm.supabase.co")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_conn():
    """Get database connection."""
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

# ==================== EXPLORE PAGE API ====================

def api_explore_trending():
    """Get trending books for explore page."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        sort_by = request.args.get('sort', 'rating')
        
        # Calculate offset
        offset = (page - 1) * limit
        
        conn = get_db_conn()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        try:
            with conn.cursor() as cur:
                # Build sort clause
                sort_clause = ""
                if sort_by == 'rating':
                    sort_clause = "ORDER BY average_rating DESC, ratings_count DESC"
                elif sort_by == 'popularity':
                    sort_clause = "ORDER BY ratings_count DESC, average_rating DESC"
                elif sort_by == 'newest':
                    sort_clause = "ORDER BY published_year DESC"
                else:
                    sort_clause = "ORDER BY average_rating DESC, ratings_count DESC"
                
                # Get total count for pagination
                cur.execute("""
                    SELECT COUNT(*) FROM books 
                    WHERE average_rating IS NOT NULL 
                    AND ratings_count IS NOT NULL 
                    AND ratings_count > 0
                """)
                total_books = cur.fetchone()[0]
                
                # Get trending books
                cur.execute(f"""
                    SELECT isbn13, title, authors, description, thumbnail, 
                           average_rating, ratings_count, num_pages, published_year,
                           categories, simple_categories
                    FROM books 
                    WHERE average_rating IS NOT NULL 
                    AND ratings_count IS NOT NULL 
                    AND ratings_count > 0
                    {sort_clause}
                    LIMIT %s OFFSET %s
                """, (limit, offset))
                
                books_data = cur.fetchall()
                books = []
                
                for book in books_data:
                    books.append({
                        'isbn13': book[0],
                        'title': book[1],
                        'authors': book[2],
                        'description': book[3] or 'Không có mô tả',
                        'image_url': book[4],
                        'average_rating': float(book[5]) if book[5] else 0.0,
                        'ratings_count': book[6] or 0,
                        'num_pages': book[7],
                        'published_year': book[8],
                        'categories': book[9],
                        'simple_categories': book[10]
                    })
                
                # Calculate pagination
                total_pages = math.ceil(total_books / limit)
                
                pagination = {
                    'current_page': page,
                    'total_pages': total_pages,
                    'total_books': total_books,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                }
                
                return jsonify({
                    'success': True,
                    'books': books,
                    'pagination': pagination
                })
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ Error getting trending books: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Failed to get trending books'}), 500

# ==================== CATEGORIES API ====================

def api_get_categories():
    """Get all available categories."""
    try:
        conn = get_db_conn()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        try:
            with conn.cursor() as cur:
                # Get categories with book counts
                cur.execute("""
                    SELECT simple_categories, COUNT(*) as book_count
                    FROM books 
                    WHERE simple_categories IS NOT NULL 
                    AND simple_categories != ''
                    GROUP BY simple_categories
                    ORDER BY book_count DESC
                """)
                
                categories_data = cur.fetchall()
                categories = []
                
                # Category mappings
                category_mapping = {
                    'Fiction': {'slug': 'van-hoc', 'name': 'Văn học', 'icon': '📖'},
                    'Science': {'slug': 'khoa-hoc-vien-tuong', 'name': 'Khoa học - Viễn tưởng', 'icon': '🔬'},
                    'History': {'slug': 'lich-su', 'name': 'Lịch sử', 'icon': '📜'},
                    'Business': {'slug': 'kinh-doanh', 'name': 'Kinh doanh', 'icon': '💼'},
                    'Psychology': {'slug': 'tam-ly', 'name': 'Tâm lý', 'icon': '🧠'},
                    'Mystery': {'slug': 'trinh-tham', 'name': 'Trinh thám', 'icon': '🔍'},
                    'Sports': {'slug': 'the-thao', 'name': 'Thể thao', 'icon': '⚽'},
                    'Travel': {'slug': 'du-lich', 'name': 'Du lịch', 'icon': '✈️'},
                    'Cooking': {'slug': 'nau-an', 'name': 'Nấu ăn', 'icon': '🍳'},
                    'Education': {'slug': 'giao-duc', 'name': 'Giáo dục', 'icon': '🎓'},
                    'Technology': {'slug': 'cong-nghe', 'name': 'Công nghệ', 'icon': '💻'},
                    'Health': {'slug': 'suc-khoe', 'name': 'Sức khỏe', 'icon': '❤️'},
                    'Family': {'slug': 'gia-dinh', 'name': 'Gia đình', 'icon': '👨‍👩‍👧‍👦'},
                    'Children': {'slug': 'thieu-nhi', 'name': 'Thiếu nhi', 'icon': '🧸'}
                }
                
                for cat_data in categories_data:
                    category_name = cat_data[0]
                    book_count = cat_data[1]
                    
                    if category_name in category_mapping:
                        cat_info = category_mapping[category_name]
                        categories.append({
                            'slug': cat_info['slug'],
                            'name': cat_info['name'],
                            'icon': cat_info['icon'],
                            'book_count': book_count,
                            'original_name': category_name
                        })
                
                return jsonify({
                    'success': True,
                    'categories': categories
                })
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ Error getting categories: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Failed to get categories'}), 500

def api_get_category_books(category_slug):
    """Get books by category."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 12, type=int)
        sort_by = request.args.get('sort', 'popularity')
        search = request.args.get('search', '')
        min_rating = request.args.get('min_rating', type=float)
        
        # Calculate offset
        offset = (page - 1) * limit
        
        # Map category slug to database category
        category_mapping = {
            'van-hoc': 'Fiction',
            'khoa-hoc-vien-tuong': 'Science',
            'lich-su': 'History',
            'kinh-doanh': 'Business',
            'tam-ly': 'Psychology',
            'trinh-tham': 'Mystery',
            'the-thao': 'Sports',
            'du-lich': 'Travel',
            'nau-an': 'Cooking',
            'giao-duc': 'Education',
            'cong-nghe': 'Technology',
            'suc-khoe': 'Health',
            'gia-dinh': 'Family',
            'thieu-nhi': 'Children'
        }
        
        db_category = category_mapping.get(category_slug)
        if not db_category:
            return jsonify({'success': False, 'error': 'Category not found'}), 404
        
        conn = get_db_conn()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        try:
            with conn.cursor() as cur:
                # Build WHERE clause
                where_conditions = ["simple_categories = %s"]
                params = [db_category]
                
                if search:
                    where_conditions.append("(LOWER(title) LIKE LOWER(%s) OR LOWER(authors) LIKE LOWER(%s))")
                    params.extend([f'%{search}%', f'%{search}%'])
                
                if min_rating:
                    where_conditions.append("average_rating >= %s")
                    params.append(min_rating)
                
                where_clause = " AND ".join(where_conditions)
                
                # Build sort clause
                sort_clause = ""
                if sort_by == 'popularity':
                    sort_clause = "ORDER BY ratings_count DESC, average_rating DESC"
                elif sort_by == 'rating':
                    sort_clause = "ORDER BY average_rating DESC, ratings_count DESC"
                elif sort_by == 'newest':
                    sort_clause = "ORDER BY published_year DESC"
                elif sort_by == 'title':
                    sort_clause = "ORDER BY title ASC"
                else:
                    sort_clause = "ORDER BY ratings_count DESC, average_rating DESC"
                
                # Get total count
                cur.execute(f"""
                    SELECT COUNT(*) FROM books 
                    WHERE {where_clause}
                """, params)
                total_books = cur.fetchone()[0]
                
                # Get books
                cur.execute(f"""
                    SELECT isbn13, title, authors, description, thumbnail, 
                           average_rating, ratings_count, num_pages, published_year
                    FROM books 
                    WHERE {where_clause}
                    {sort_clause}
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])
                
                books_data = cur.fetchall()
                books = []
                
                for i, book in enumerate(books_data):
                    books.append({
                        'id': book[0],  # Using ISBN13 as ID
                        'isbn13': book[0],
                        'title': book[1],
                        'author': book[2],
                        'description': book[3] or 'Không có mô tả',
                        'image_url': book[4],
                        'average_rating': float(book[5]) if book[5] else 0.0,
                        'ratings_count': book[6] or 0,
                        'num_pages': book[7],
                        'published_year': book[8]
                    })
                
                # Calculate pagination
                total_pages = math.ceil(total_books / limit)
                
                return jsonify({
                    'success': True,
                    'books': books,
                    'total_books': total_books,
                    'current_page': page,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                })
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ Error getting category books: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Failed to get category books'}), 500

# ==================== BESTSELLERS API ====================

def api_get_bestsellers():
    """Get bestseller books."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 12, type=int)
        period = request.args.get('period', 'all')
        sort_by = request.args.get('sort', 'popularity')
        search = request.args.get('search', '')
        category = request.args.get('category', '')
        
        # Calculate offset
        offset = (page - 1) * limit
        
        conn = get_db_conn()
        if not conn:
            return jsonify({'success': False, 'error': 'Database connection failed'}), 500
        
        try:
            with conn.cursor() as cur:
                # Build WHERE clause
                where_conditions = ["ratings_count IS NOT NULL", "ratings_count > 10"]
                params = []
                
                if search:
                    where_conditions.append("(LOWER(title) LIKE LOWER(%s) OR LOWER(authors) LIKE LOWER(%s))")
                    params.extend([f'%{search}%', f'%{search}%'])
                
                if category:
                    # Map category slug to database category
                    category_mapping = {
                        'van-hoc': 'Fiction',
                        'kinh-te': 'Business',
                        'tam-ly': 'Psychology',
                        'khoa-hoc-vien-tuong': 'Science',
                        'lich-su': 'History',
                        'trinh-tham': 'Mystery'
                    }
                    db_category = category_mapping.get(category)
                    if db_category:
                        where_conditions.append("simple_categories = %s")
                        params.append(db_category)
                
                where_clause = " AND ".join(where_conditions)
                
                # Build sort clause based on sort_by
                sort_clause = ""
                if sort_by == 'popularity':
                    sort_clause = "ORDER BY ratings_count DESC, average_rating DESC"
                elif sort_by == 'rating':
                    sort_clause = "ORDER BY average_rating DESC, ratings_count DESC"
                elif sort_by == 'views':
                    # For now, use ratings_count as proxy for views
                    sort_clause = "ORDER BY ratings_count DESC"
                elif sort_by == 'recent':
                    sort_clause = "ORDER BY published_year DESC, ratings_count DESC"
                else:
                    sort_clause = "ORDER BY ratings_count DESC, average_rating DESC"
                
                # Get total count
                cur.execute(f"""
                    SELECT COUNT(*) FROM books 
                    WHERE {where_clause}
                """, params)
                total_books = cur.fetchone()[0]
                
                # Get bestseller books
                cur.execute(f"""
                    SELECT isbn13, title, authors, description, thumbnail, 
                           average_rating, ratings_count, num_pages, published_year,
                           simple_categories
                    FROM books 
                    WHERE {where_clause}
                    {sort_clause}
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])
                
                books_data = cur.fetchall()
                books = []
                
                for i, book in enumerate(books_data):
                    rank = offset + i + 1
                    books.append({
                        'id': book[0],  # Using ISBN13 as ID
                        'isbn13': book[0],
                        'title': book[1],
                        'author': book[2],
                        'description': book[3] or 'Không có mô tả',
                        'image_url': book[4],
                        'average_rating': float(book[5]) if book[5] else 0.0,
                        'ratings_count': book[6] or 0,
                        'num_pages': book[7],
                        'published_year': book[8],
                        'category': book[9],
                        'rank': rank,
                        'views': book[6] or 0,  # Using ratings_count as proxy for views
                        'sales': book[6] or 0   # Using ratings_count as proxy for sales
                    })
                
                # Calculate pagination
                total_pages = math.ceil(total_books / limit)
                
                return jsonify({
                    'success': True,
                    'books': books,
                    'total_books': total_books,
                    'current_page': page,
                    'total_pages': total_pages,
                    'has_next': page < total_pages,
                    'has_prev': page > 1
                })
                
        finally:
            conn.close()
            
    except Exception as e:
        print(f"❌ Error getting bestsellers: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Failed to get bestsellers'}), 500