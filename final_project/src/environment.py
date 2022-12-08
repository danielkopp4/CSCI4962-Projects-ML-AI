import gym
import pydealer
import numpy as np


# observation: player sum, cards shown: vector of 52 card seen or not seen, dealer top card
# later: allow other players to play too, but only have the bot go first just treat them as useless

# actions hit, stand

# return the union of a, b
def card_union(a, b):
    return a.union(b)

value_mapping = {
    "king": 10,
    "queen": 10,
    "jack": 10,
    "ace": 11   # for now
}

def value(card):
    val = card.value.lower()
    try:
        val = int(val)
        return val
    except ValueError:
        return value_mapping[val]

suit_mapping = {
    "hearts": 0,
    "spades": 1,
    "diamonds": 2,
    "clubs": 3
}

def suit_id(suit):
    return suit_mapping[suit.lower()]

card_id_mapping = {
    "king": 9,
    "queen": 10,
    "jack": 11,
    "ace": 12        
}

def card_id(card_type):
    try:
        amnt = int(card_type)
        return amnt - 2
    except ValueError:
        return card_id_mapping[card_type.lower()]
        

def id(card):
    return 13 * suit_id(card.suit) + card_id(card.value)

def id_vec(cards):
    return [id(card) for card in cards]

def add_cards(stack):
    amount = 0 
    for card in stack:
        amount += value(card)

    return amount


WIN = "win"
CONTINUE = "continue"
LOSE = "lose"
PUSH = "push"

reward_multiplier = 1

rewards = {
    WIN: 1,
    LOSE: -1,
    PUSH: 0,
    CONTINUE: 0
}

# have to handle ace condition later
class BlackJackGame:
    def __init__(self, infinite=False):
        self.infinite = infinite
        self.reset_game()

    def reset_game(self):
        # reset deck and shuffle
        self.deck = pydealer.Deck()
        self.deck.shuffle()

        self._seen_cards = set()
        self.reset_round()

    @property
    def seen_cards(self):
        return self._seen_cards

    @property
    def player_sum(self):
        return self._player_sum

    def dealer_turn(self):
        # get dealer cards
        # rules are to add until hits 17 then stand
        dealer_sum = add_cards(self.dealer)

        # logic will be that if sum > 21 && viable ace -10
        while dealer_sum < 17:
            new_card = self.deck.deal()
            self.dealer = card_union(self.dealer, new_card)
            dealer_sum += value(new_card[0])

        self._seen_cards = card_union(self._seen_cards, id_vec(self.dealer))

        return dealer_sum

    def reset_round(self):
        # give dealer cards
        self.dealer = set(self.deck.deal(2))

        # give player cards
        self.player = set(self.deck.deal(2))
        self._player_sum = add_cards(self.player)


        # give other player cards (opt)

        # calc seen_cards
        self._seen_cards = card_union(self._seen_cards, id_vec(self.player))
        self._seen_cards = card_union(self._seen_cards, id_vec(self.dealer))



    def hit(self):
        new_card = self.deck.deal(1)
        self._seen_cards = card_union(self._seen_cards, id_vec(new_card))
        self.player = card_union(self.player, new_card)
        self._player_sum += value(new_card[0])

        if self.player_sum > 21:
            # bust
            self.reset_round()
            return LOSE

        return CONTINUE


    def stand(self):
        dealer_sum = self.dealer_turn()

        return_val = WIN

        if dealer_sum == self.player_sum or dealer_sum > 21:
            return_val = PUSH
        elif dealer_sum > self.player_sum:
            return_val = LOSE
        else:
            return_val = WIN

        self.reset_round()

        return return_val

    @property
    def has_ace(self):
        for card in self.player:
            if card.value.lower() == "ace":
                return True
        return False

    @property
    def ended(self):
        if self.infinite:
            return True
        
        return len(self.deck) < 0.5 * 52

    @property
    def public_dealer(self):
        return next(iter(self.dealer))

def map_seen_to_fixed_arr(seen_cards):
    out = np.zeros((52,))
    out[list(seen_cards)] = 1
    return out

class BlackJackEnv(gym.Env):
    def __init__(self, infinite=False):
        self.game = BlackJackGame(infinite)
        self.action_space = gym.spaces.Discrete(2) # hit, stand
        low = np.zeros((3 + 52,))
        high = np.concatenate(([21, 11], np.ones((1 + 52))))
        self.observation_space = gym.spaces.Box(
            low=low, 
            high=high, 
            shape=(3 + 52,), 
            dtype=np.float64
        )

    def step(self, action):
        if action == 0:
            result = self.game.hit()
        elif action == 1:
            result = self.game.stand()
        else:
            raise ValueError(f"unknown action '{action}'")

        reward = rewards[result] * reward_multiplier
        observation = self._calculate_observation()
        done = self.game.ended
        info = {}

        return observation, reward, done, info

    def _calculate_observation(self):
        return np.concatenate((
            np.array([self.game.player_sum, value(self.game.public_dealer), float(self.game.has_ace)]),
            map_seen_to_fixed_arr(self.game.seen_cards)
        ))

    def reset(self):
        self.game.reset_game()
        return self._calculate_observation()

    def render(self, mode=''):
        
        if mode == "console":
            def card_formatter(card):
                return f"({card.suit};{card.value})"

            def format_cards(cards):
                return [card_formatter(card) for card in cards]

            # print player cards
            print(f"player cards: {format_cards(self.game.player)}; sum: {self.game.player_sum}; ",end="")
            print(f"dealer top card: {card_formatter(self.game.public_dealer)}")

    def close(self):
        pass

