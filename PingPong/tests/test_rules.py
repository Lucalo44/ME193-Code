from game.rules import Match


def play(m, sequence):
    results = []
    for who in sequence:
        results.append(m.award_point(who))
    return results


def test_game_to_11():
    m = Match()
    r = play(m, ["player"] * 11)
    assert r[-1].game_over and r[-1].game_winner == "player"
    assert not any(x.game_over for x in r[:-1])
    assert m.games["player"] == 1


def test_win_by_two_at_deuce():
    m = Match()
    play(m, ["player", "cpu"] * 10)          # 10-10
    r = m.award_point("player")              # 11-10: not over
    assert not r.game_over
    r = m.award_point("cpu")                 # 11-11
    assert not r.game_over
    play(m, ["cpu"])                         # 11-12
    r = m.award_point("cpu")                 # 11-13
    assert r.game_over and r.game_winner == "cpu"


def test_serve_alternates_every_two_points():
    m = Match(first_server="cpu")
    servers = []
    for _ in range(8):
        servers.append(m.server)
        m.award_point("player" if len(servers) % 3 else "cpu")
    assert servers == ["cpu", "cpu", "player", "player", "cpu", "cpu", "player", "player"]


def test_serve_alternates_every_point_at_deuce():
    m = Match(first_server="cpu")
    play(m, ["player", "cpu"] * 10)          # 10-10
    servers = []
    for who in ["player", "cpu", "player", "cpu"]:
        servers.append(m.server)
        m.award_point(who)
    assert servers[0] != servers[1] and servers[1] != servers[2] and servers[2] != servers[3]


def test_first_server_alternates_between_games():
    m = Match(first_server="cpu")
    play(m, ["cpu"] * 11)
    m.start_next_game()
    assert m.server == "player"
    assert m.points == {"player": 0, "cpu": 0}


def test_best_of_three_match():
    m = Match(games_per_match=3)
    play(m, ["player"] * 11)
    m.start_next_game()
    play(m, ["cpu"] * 11)
    m.start_next_game()
    r = play(m, ["player"] * 11)[-1]
    assert r.match_over and r.match_winner == "player"
    assert m.games == {"player": 2, "cpu": 1}


def test_award_game_penalty():
    m = Match()
    play(m, ["player"] * 5)
    r = m.award_game("cpu")
    assert r.game_over and r.game_winner == "cpu"
    assert m.games["cpu"] == 1


def test_no_points_after_match_over():
    m = Match(games_per_match=1)
    play(m, ["cpu"] * 11)
    r = m.award_point("player")
    assert r.match_over and m.points["player"] == 0
