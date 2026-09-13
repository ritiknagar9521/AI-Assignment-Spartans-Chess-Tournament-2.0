import time
from config import *


INF = 10**9
MATE_SCORE = 100000


class B23EE1061:
    """Iterative-deepening Negamax player with alpha-beta pruning."""

    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 1

        # Search settings
        self.max_depth = 7
        self.search_time = 0.65
        self.start_time = 0.0
        self.deadline = 0.0
        self.moves_played = 0
        self.game_start_time = None

        # Position -> (searched_depth, score, flag)
        self.transposition_table = {}

    
    # Main move selection
  

    def get_best_move(self):
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Keep a safety margin under the 60 second tournament clock.
        if self.game_start_time is None:
            self.game_start_time = time.time()

        self.start_time = time.time()
        self.search_time = self._choose_time()
        self.deadline = self.start_time + self.search_time
        self.nodes_expanded = 0

        # Always have a legal fallback move.
        best_move = self._order_moves(legal_moves)[0]
        best_score = -INF

        # Iterative deepening: only keep results from completed depths.
        for depth in range(1, self.max_depth + 1):
            self.depth = depth

            try:
                score, move = self._search_root(legal_moves, depth)
            except TimeoutError:
                break

            if move is not None:
                best_move = move
                best_score = score

            # No need to search deeper after a forced mate.
            if abs(best_score) >= MATE_SCORE - 1000:
                break

        self.moves_played += 1
        return best_move

    def _choose_time(self):
        """Give each move a small safe slice of the total game clock."""
        elapsed = time.time() - self.game_start_time
        remaining = max(1.0, 52.0 - elapsed)
        moves_left = max(1, 75 - self.moves_played)

        legal_count = len(self.engine.get_legal_moves())
        time_for_move = remaining / moves_left

        # More choices -> position is usually more difficult.
        if legal_count >= 12:
            time_for_move *= 1.25
        elif legal_count <= 4:
            time_for_move *= 0.75

        return max(0.08, min(time_for_move, 1.2, remaining * 0.20))

    # Alpha-Beta / Negamax

    def _search_root(self, legal_moves, depth):
        best_score = -INF
        best_move = None
        alpha = -INF
        beta = INF

        ordered_moves = self._order_moves(legal_moves)
        current_color = 1 if self.engine.white_to_move else -1

        for move in ordered_moves:
            self._check_time()

            self.engine.make_move(move)
            try:
                # After White moves, Black is to move, so color = -1.
                # After Black moves, White is to move, so color = +1.
                next_color = -current_color
                score = -self._negamax(depth - 1, -beta, -alpha, next_color)
            finally:
                self.engine.undo_move()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        return best_score, best_move

    def _negamax(self, depth, alpha, beta, color):
        self.nodes_expanded += 1

        if self.nodes_expanded % 128 == 0:
            self._check_time()

        # Terminal positions are checked before the depth cutoff.
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            if self.engine.is_in_check():
                # Current side has been checkmated.
                return -MATE_SCORE - depth
            return 0

        # At the depth limit, continue with tactical captures.
        if depth <= 0:
            return self._quiescence(alpha, beta, color, 3)

        # Check transposition table.
        key = self._position_key()
        stored = self.transposition_table.get(key)
        if stored is not None:
            stored_depth, stored_score, stored_flag = stored
            if stored_depth >= depth:
                if stored_flag == 'EXACT':
                    return stored_score
                if stored_flag == 'LOWER' and stored_score >= beta:
                    return stored_score
                if stored_flag == 'UPPER' and stored_score <= alpha:
                    return stored_score

        original_alpha = alpha
        best_score = -INF
        cutoff = False

        for move in self._order_moves(legal_moves):
            self.engine.make_move(move)
            try:
                score = -self._negamax(depth - 1, -beta, -alpha, -color)
            finally:
                self.engine.undo_move()

            if score > best_score:
                best_score = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                cutoff = True
                break

        # Store a standard alpha-beta bound.
        if cutoff:
            flag = 'LOWER'
        elif best_score <= original_alpha:
            flag = 'UPPER'
        else:
            flag = 'EXACT'

        self._store(key, depth, best_score, flag)
        return best_score


    # Quiescent Search

    def _quiescence(self, alpha, beta, color, depth):
        """Continue tactical captures instead of stopping abruptly."""
        self.nodes_expanded += 1

        if self.nodes_expanded % 128 == 0:
            self._check_time()

        stand_pat = color * self.evaluate_board()

        if stand_pat >= beta:
            return stand_pat

        if stand_pat > alpha:
            alpha = stand_pat

        if depth <= 0:
            return alpha

        legal_moves = self.engine.get_legal_moves()
        in_check = self.engine.is_in_check()

        # If in check, all legal evasions must be considered.
        # Otherwise, only captures are useful for quiescence search.
        if in_check:
            tactical_moves = legal_moves
        else:
            tactical_moves = [
                move for move in legal_moves
                if move.piece_captured != EMPTY_SQUARE
            ]

        for move in self._order_moves(tactical_moves):
            self.engine.make_move(move)
            try:
                score = -self._quiescence(
                    -beta, -alpha, -color, depth - 1
                )
            finally:
                self.engine.undo_move()

            if score >= beta:
                return score

            if score > alpha:
                alpha = score

        return alpha

    # Move Ordering

    def _order_moves(self, moves):
        """Captures first using a simple MVV-LVA score."""
        def move_score(move):
            if move.piece_captured == EMPTY_SQUARE:
                return 0

            victim = abs(PIECE_VALUES.get(move.piece_captured, 0))
            attacker = abs(PIECE_VALUES.get(move.piece_moved, 1))
            return 10000 + victim * 100 - attacker

        return sorted(moves, key=move_score, reverse=True)

    # Evaluation Function

    def evaluate_board(self):
        """
        Evaluation from White's perspective.

        Uses simple features that are easy to explain in a viva:
            1. Material
            2. Piece-square tables
            3. Mobility
            4. Check bonus
            5. Basic king activity in endgames
        """
        state = self.engine.get_game_state()

        if state == "checkmate":
            # The side to move has lost.
            return -MATE_SCORE if self.engine.white_to_move else MATE_SCORE

        if state == "stalemate":
            return 0

        score = 0
        white_material = 0
        black_material = 0

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.engine.board[r][c]

                if piece == EMPTY_SQUARE:
                    continue

                score += PIECE_VALUES[piece]

                value = abs(PIECE_VALUES[piece])
                if piece[0] == 'w':
                    white_material += value
                else:
                    black_material += value

                # Use the supplied positional tables.
                row = r if piece[0] == 'w' else BOARD_HEIGHT - 1 - r
                sign = 1 if piece[0] == 'w' else -1

                if piece[1] == 'P':
                    score += sign * PAWN_PST[row][c]
                elif piece[1] == 'N':
                    score += sign * KNIGHT_PST[row][c]
                elif piece[1] == 'B':
                    score += sign * BISHOP_PST[row][c]
                elif piece[1] == 'R':
                    score += sign * ROOK_PST[row][c]

        # Mobility is a small bonus, so it does not overpower material.
        mobility = len(self.engine.get_legal_moves())
        if self.engine.white_to_move:
            score += mobility
        else:
            score -= mobility

        # The assignment awards +2 for giving check.
        if self.engine.is_in_check():
            if self.engine.white_to_move:
                score -= 2
            else:
                score += 2

        # In an endgame, active kings are more useful.
        if white_material + black_material <= 500:
            score += self._king_activity('w')
            score -= self._king_activity('b')

        return score

    def _king_activity(self, color):
        """Small endgame bonus for moving the king toward the centre."""
        king_piece = WHITE_KING if color == 'w' else BLACK_KING

        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                if self.engine.board[r][c] == king_piece:
                    return 8 - abs(3 - r) - abs(2 - c)

        return 0

    # ================================================================
    # Helpers
    # ================================================================

    def _position_key(self):
        return (
            tuple(tuple(row) for row in self.engine.board),
            self.engine.white_to_move
        )

    def _store(self, key, depth, score, flag):
        # Keep the table bounded so memory cannot grow forever.
        if len(self.transposition_table) >= 30000:
            self.transposition_table.clear()

        self.transposition_table[key] = (depth, score, flag)

    def _check_time(self):
        if time.time() >= self.deadline:
            raise TimeoutError
