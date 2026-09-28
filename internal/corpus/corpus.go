// Package data holds the synthetic Bangalore restaurant corpus that
// MenuMind's search index operate on. Everything is generated once at
// startup from hand-authored seeds with deterministic variation, so the
// dataset is stable across restarts and testable.
package corpus

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
)

// Area is a delivery zone with a representative coordinate.
type Area struct {
	Name string
	Lat  float64
	Lng  float64
}

// Restaurant is one listings row on the marketplace.
type Restaurant struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Cuisines    []string `json:"cuisines"`
	Area        string   `json:"area"`
	Rating      float64  `json:"rating"`
	Ratings     int      `json:"ratings"` // number of user ratings
	PriceForTwo int      `json:"price_for_two"` // rupees
	VegOnly     bool     `json:"veg_only"`
	PrepMin     int      `json:"prep_min"` // kitchen prep time, minutes
	Lat         float64  `json:"lat"`
	Lng         float64  `json:"lng"`
}

// MenuItem is one searchable dish.
type MenuItem struct {
	ID          string  `json:"id"`
	RestID      string  `json:"restaurant_id"`
	RestName    string  `json:"restaurant_name"`
	Name        string  `json:"name"`
	Description string  `json:"description"`
	Category    string  `json:"category"`
	Cuisine     string  `json:"cuisine"`
	Area        string  `json:"area"`
	Price       int     `json:"price"` // rupees
	Veg         bool    `json:"veg"`
	Spice       int     `json:"spice"` // 0-3
	Calories    int     `json:"calories"`
	Rating      float64 `json:"rating"`
	Bestseller  bool    `json:"bestseller"`
}

// Text is what gets embedded and BM25-indexed for this dish.
func (it MenuItem) Text() string {
	return it.Name + ". " + it.Description + " " + it.Category + " " +
		strings.ReplaceAll(it.Cuisine, ",", " ") + " " + it.RestName + " " + it.Area
}

type seedRest struct {
	name     string
	area     string
	cuisines []string
	price    int
	rating   float64
	vegOnly  bool
}

var Areas = map[string]Area{
	"Koramangala":  {Name: "Koramangala", Lat: 12.9345, Lng: 77.6266},
	"Indiranagar":  {Name: "Indiranagar", Lat: 12.9784, Lng: 77.6408},
	"HSR Layout":   {Name: "HSR Layout", Lat: 12.9116, Lng: 77.6389},
	"Whitefield":   {Name: "Whitefield", Lat: 12.9698, Lng: 77.7500},
	"Jayanagar":    {Name: "Jayanagar", Lat: 12.9299, Lng: 77.5826},
	"JP Nagar":     {Name: "JP Nagar", Lat: 12.9063, Lng: 77.5857},
	"Marathahalli": {Name: "Marathahalli", Lat: 12.9569, Lng: 77.7011},
	"Church Street": {Name: "Church Street", Lat: 12.9740, Lng: 77.6045},
}

var seedRestaurants = []seedRest{
	{"Ruchi Andhra Kitchen", "Koramangala", []string{"andhra", "biryani"}, 400, 4.5, false},
	{"Dosa Theory", "Jayanagar", []string{"southindian"}, 250, 4.6, true},
	{"Biryani Blues", "HSR Layout", []string{"biryani", "kebab"}, 450, 4.2, false},
	{"Wok This Way", "Indiranagar", []string{"chinese"}, 350, 4.1, false},
	{"The Cheesy Crust", "Koramangala", []string{"pizza"}, 500, 4.4, false},
	{"Grill & Chill", "Church Street", []string{"burgers", "continental"}, 550, 4.3, false},
	{"Sweet Symphony", "Indiranagar", []string{"desserts", "beverages"}, 300, 4.7, true},
	{"Coastal Canvas", "Whitefield", []string{"seafood", "andhra"}, 600, 4.4, false},
	{"Tandoori Tales", "Marathahalli", []string{"kebab", "northindian"}, 500, 4.0, false},
	{"Punjab Grill House", "JP Nagar", []string{"northindian"}, 450, 4.3, false},
	{"Thai Thani", "Church Street", []string{"thai"}, 550, 4.2, false},
	{"Mysuru Tiffin Room", "Jayanagar", []string{"southindian", "beverages"}, 200, 4.5, true},
	{"Spice Junction", "HSR Layout", []string{"northindian", "chinese"}, 400, 3.9, false},
	{"Gongura Go!", "Marathahalli", []string{"andhra"}, 350, 4.1, false},
	{"Noodle Nest", "Whitefield", []string{"chinese", "thai"}, 400, 3.8, false},
	{"Urban Brunch Co.", "Indiranagar", []string{"continental", "beverages"}, 650, 4.5, false},
	{"Kebab Karavan", "Koramangala", []string{"kebab", "biryani"}, 500, 4.4, false},
	{"Dal Baati Junction", "Whitefield", []string{"northindian"}, 400, 4.0, true},
	{"Espresso Express", "Church Street", []string{"beverages", "desserts"}, 350, 4.6, false},
	{"Chennai Chat Corner", "HSR Layout", []string{"southindian"}, 250, 4.0, true},
	{"Flame & Fork", "JP Nagar", []string{"continental", "kebab"}, 600, 4.2, false},
	{"Bonda Bite", "Marathahalli", []string{"southindian", "beverages"}, 220, 4.1, true},
	{"curryleaf.", "Church Street", []string{"northindian", "biryani"}, 480, 4.5, false},
	{"Sri Sweets & Snacks", "Jayanagar", []string{"desserts", "southindian"}, 280, 4.6, true},
	{"Basil & Burnt Garlic", "Whitefield", []string{"thai", "chinese"}, 520, 4.3, true},
	{"The Burger Bureau", "Koramangala", []string{"burgers"}, 400, 4.2, false},
	{"Amma's Andhra Meals", "JP Nagar", []string{"andhra"}, 300, 4.4, false},
	{"Pizza Postale", "Indiranagar", []string{"pizza"}, 550, 4.5, false},
	{"Wok & Roll", "HSR Layout", []string{"chinese"}, 380, 3.9, false},
	{"Coast to Coast", "Church Street", []string{"seafood"}, 700, 4.6, false},
	{"Halwa Heaven", "Marathahalli", []string{"desserts"}, 320, 4.5, true},
	{"Roti Roll Co.", "Whitefield", []string{"kebab", "northindian"}, 360, 4.0, false},
	{"Saffron Story", "Koramangala", []string{"biryani", "northindian"}, 520, 4.6, false},
	{"Green Bowl", "Indiranagar", []string{"continental"}, 450, 4.4, true},
	{"Chutney Chang", "Jayanagar", []string{"chinese", "northindian"}, 420, 4.1, false},
	{"Filter Kaapi Co.", "Church Street", []string{"beverages", "southindian"}, 240, 4.7, true},
}

type bankItem struct {
	name    string
	desc    string
	price   int
	veg     bool
	spice   int
	cal     int
	cat     string
}

// banks maps cuisine key to its item bank. The agent's search tools and the
// menu browser read from the corpus generated out of these.
var banks = map[string][]bankItem{
	"andhra": {
		{"Natu Kodi Pulusu", "Country chicken simmered in fiery tamarind gravy with garlic", 320, false, 3, 520, "Main Course"},
		{"Gongura Chicken Curry", "Sorrel-leaf chicken, tangy Andhra classic with dry red chilli", 340, false, 3, 540, "Main Course"},
		{"Andhra Chili Chicken", "Boneless chicken tossed with Guntur chilli and curry leaf", 290, false, 3, 480, "Starter"},
		{"Royyala Iguru", "Prawn masala roasted with coconut and shallots", 380, false, 2, 450, "Main Course"},
		{"Rayalaseema Ragi Sangati", "Steamed ragi balls served with spicy country chicken", 260, false, 3, 490, "Main Course"},
		{"Bendakaya Vepudu", "Crisp okra fry with peanuts and sesame", 220, true, 2, 320, "Side"},
		{"Pappu Charu", "Toor dal broth with tomato, garlic and coriander", 160, true, 1, 210, "Side"},
		{"Avakaya Biryani", "Biryani layered with raw mango pickle masala", 310, false, 3, 610, "Biryani"},
	},
	"biryani": {
		{"Hyderabadi Chicken Dum Biryani", "Long-grain basmati sealed and slow-cooked with saffron and mint", 320, false, 2, 680, "Biryani"},
		{"Guntur Chicken Biryani", "Extra-hot Guntur chilli biryani with boiled egg", 340, false, 3, 700, "Biryani"},
		{"Mutton Biryani", "Tender mutton chunks, dum-style with fried onions", 420, false, 2, 740, "Biryani"},
		{"Veg Dum Biryani", "Garden vegetables and paneer layered with saffron rice", 240, true, 1, 560, "Biryani"},
		{"Paneer Tikka Biryani", "Charred paneer tikka folded into mildly spiced biryani", 280, true, 2, 620, "Biryani"},
		{"Egg Biryani", "Masala-fried eggs with biryani rice and raita", 220, false, 2, 540, "Biryani"},
		{"Chicken 65 Biryani", "Biryani topped with fiery chicken 65 bits", 330, false, 3, 690, "Biryani"},
		{"Double Ka Meetha", "Hyderabadi bread pudding with saffron and dry fruits", 140, true, 0, 420, "Dessert"},
	},
	"southindian": {
		{"Masala Dosa", "Crisp dosa with potato masala, sambar and two chutneys", 120, true, 1, 340, "Dosa"},
		{"Ghee Roast Dosa", "Paper-thin dosa roasted in pure ghee", 150, true, 1, 380, "Dosa"},
		{"Mysore Masala Dosa", "Spicy red garlic chutney layered dosa", 160, true, 2, 410, "Dosa"},
		{"Rava Onion Dosa", "Lacy semolina dosa with onion and green chilli", 130, true, 1, 350, "Dosa"},
		{"Idli Vada Sambar", "Two steamed idlis and a medu vada with sambar", 90, true, 1, 280, "Tiffin"},
		{"Medu Vada", "Crisp urad dal vadas with coconut chutney", 80, true, 1, 240, "Tiffin"},
		{"Upma Kesari Bath", "Chow chow bath: savory upma with sweet kesari", 100, true, 0, 380, "Tiffin"},
		{"Bisi Bele Bath", "Rice, dal and vegetables cooked with tamarind spice", 140, true, 2, 420, "Rice"},
		{"Curd Rice with Pickle", "Comfort curd rice tempered with mustard and curry leaf", 110, true, 0, 300, "Rice"},
		{"Filter Coffee", "Traditional South Indian filter kaapi, frothy", 60, true, 0, 90, "Beverage"},
	},
	"northindian": {
		{"Dal Makhani", "Black urad dal simmered overnight with butter and cream", 240, true, 1, 430, "Main Course"},
		{"Butter Chicken", "Tandoori chicken in silky tomato-makhani gravy", 340, false, 1, 560, "Main Course"},
		{"Paneer Butter Masala", "Cottage cheese in rich cashew-tomato gravy", 280, true, 1, 520, "Main Course"},
		{"Kadhai Paneer", "Paneer and peppers tossed in kadhai masala", 270, true, 2, 490, "Main Course"},
		{"Chole Bhature", "Punjabi chickpea curry with fluffy fried bhature", 180, true, 2, 640, "Main Course"},
		{"Rajasthani Dal Baati", "Baked wheat baati with panchmel dal and ghee", 260, true, 2, 580, "Main Course"},
		{"Jeera Rice", "Basmati rice tempered with roasted cumin", 130, true, 0, 290, "Rice"},
		{"Tandoori Roti Basket", "Four tandoor-fresh rotis with white butter", 90, true, 0, 320, "Bread"},
		{"Gulab Jamun (2 pc)", "Warm jamuns soaked in cardamom syrup", 90, true, 0, 300, "Dessert"},
	},
	"chinese": {
		{"Gobi Manchurian Gravy", "Crisp cauliflower in tangy manchurian sauce", 200, true, 2, 380, "Starter"},
		{"Veg Hakka Noodles", "Wok-tossed noodles with julienne vegetables", 180, true, 1, 460, "Noodles"},
		{"Chicken Fried Rice", "Wok-charred rice with shredded chicken", 210, false, 2, 520, "Rice"},
		{"Chilli Chicken Dry", "Indo-Chinese chilli chicken with spring onion", 240, false, 3, 480, "Starter"},
		{"Schezwan Noodles", "Fiery schezwan pepper noodles", 200, true, 3, 500, "Noodles"},
		{"Steamed Chicken Momos (6)", "Juicy steamed dumplings with fiery red chutney", 160, false, 2, 340, "Starter"},
		{"Pan-Fried Veg Momos (6)", "Crisp-bottomed momos with schezwan dip", 180, true, 2, 380, "Starter"},
		{"Sweet Corn Soup", "Creamy sweet corn and veggie soup", 120, true, 0, 160, "Soup"},
		{"Spring Rolls (4)", "Crisp rolls stuffed with cabbage and noodles", 150, true, 1, 340, "Starter"},
	},
	"pizza": {
		{"Margherita", "Classic mozzarella and basil on slow-proofed sourdough", 240, true, 0, 680, "Pizza"},
		{"Paneer Tikka Pizza", "Tandoori paneer, onion and capsicum on makhani base", 340, true, 1, 760, "Pizza"},
		{"Chicken Tikka Pizza", "Charred chicken tikka with mint drizzle", 380, false, 1, 810, "Pizza"},
		{"Veg Supreme", "Loaded with peppers, corn, olives and mushrooms", 320, true, 0, 740, "Pizza"},
		{"Pepperoni", "Double pepperoni with mozzarella and oregano", 390, false, 1, 840, "Pizza"},
		{"Garlic Bread Supreme", "Stuffed garlic bread with herb butter", 160, true, 0, 420, "Side"},
		{"Choco Lava Cake", "Molten center chocolate cake", 110, true, 0, 350, "Dessert"},
	},
	"burgers": {
		{"Classic Veg Burger", "Crunchy veg patty, lettuce and mayo in brioche", 140, true, 0, 460, "Burger"},
		{"Paneer Zinger Burger", "Crisp-fried paneer with spicy slaw", 180, true, 2, 540, "Burger"},
		{"Crispy Chicken Burger", "Buttermilk-fried chicken thigh, house sauce", 200, false, 1, 590, "Burger"},
		{"Cheese Burger", "Double cheese slice with caramelized onion", 190, true, 0, 570, "Burger"},
		{"Peri Peri Fries", "Shoestring fries dusted with peri peri", 120, true, 2, 380, "Side"},
		{"Loaded Cheese Fries", "Fries under molten cheddar and jalapeno", 160, true, 1, 520, "Side"},
		{"Chocolate Thickshake", "Belgian chocolate thickshake with cream", 150, true, 0, 480, "Beverage"},
	},
	"desserts": {
		{"Gulab Jamun (2 pc)", "Warm jamuns soaked in cardamom syrup", 90, true, 0, 300, "Dessert"},
		{"Rasmalai (2 pc)", "Soft chenna discs in saffron milk", 120, true, 0, 320, "Dessert"},
		{"Chocolate Lava Cake", "Molten center chocolate cake", 130, true, 0, 380, "Dessert"},
		{"Filter Coffee Tiramisu", "South meets Italy: kaapi-soaked tiramisu", 180, true, 0, 420, "Dessert"},
		{"Kulfi Falooda", "Malai kulfi with vermicelli, basil seeds and rose", 150, true, 0, 440, "Dessert"},
		{"Brownie with Ice Cream", "Warm fudge brownie, vanilla scoop, nuts", 170, true, 0, 510, "Dessert"},
		{"Gajar Halwa", "Slow-cooked carrot halwa with khoya and nuts", 130, true, 0, 400, "Dessert"},
		{"Mysore Pak", "Ghee-rich gram flour fudge squares", 110, true, 0, 360, "Dessert"},
	},
	"beverages": {
		{"Masala Chai", "Kadak chai brewed with ginger and cardamom", 50, true, 0, 90, "Beverage"},
		{"Cold Coffee", "Frothy cold coffee with ice cream scoop", 130, true, 0, 320, "Beverage"},
		{"Fresh Lime Soda", "Sweet-salted lime soda, ice cold", 70, true, 0, 60, "Beverage"},
		{"Mango Lassi", "Thick Alphonso mango lassi with malai", 120, true, 0, 280, "Beverage"},
		{"Rose Milk", "Chilled rose milk with basil seeds", 80, true, 0, 180, "Beverage"},
		{"Tender Coconut Water", "Straight from the shell", 70, true, 0, 45, "Beverage"},
		{"Cold Brew Tonic", "Cold brew coffee with tonic and orange peel", 160, true, 0, 70, "Beverage"},
	},
	"thai": {
		{"Pad Thai", "Stir-fried rice noodles, peanuts, lime and bean sprouts", 280, true, 1, 520, "Noodles"},
		{"Green Curry", "Coconut green curry with Thai basil and vegetables", 300, true, 2, 480, "Main Course"},
		{"Tom Yum Soup", "Hot and sour lemongrass soup with mushrooms", 220, true, 2, 160, "Soup"},
		{"Basil Fried Rice", "Thai holy basil rice with chilli and garlic", 260, true, 3, 560, "Rice"},
		{"Som Tam", "Green papaya salad, palm sugar and lime", 180, true, 2, 140, "Salad"},
	},
	"continental": {
		{"Aglio e Olio", "Spaghetti with garlic, chilli flakes and parmesan", 280, true, 1, 520, "Pasta"},
		{"Alfredo Pasta", "Fettuccine in creamy parmesan sauce", 320, true, 0, 640, "Pasta"},
		{"Grilled Chicken Salad", "Herb-grilled chicken on greens with balsamic", 300, false, 0, 380, "Salad"},
		{"Caesar Salad", "Romaine, parmesan, croutons, classic dressing", 260, true, 0, 320, "Salad"},
		{"Fish and Chips", "Beer-battered fish with tartar and thick fries", 380, false, 0, 720, "Main Course"},
		{"Mushroom Risotto", "Arborio rice with wild mushrooms and truffle oil", 360, true, 0, 610, "Main Course"},
	},
	"kebab": {
		{"Tandoori Chicken (Half)", "Clay-oven chicken marinated in hung curd and Kashmiri chilli", 300, false, 2, 520, "Kebab"},
		{"Paneer Tikka", "Char-grilled paneer with capsicum and onion", 260, true, 2, 460, "Kebab"},
		{"Seekh Kebab (4 pc)", "Minced mutton kebabs with mint chutney", 280, false, 2, 480, "Kebab"},
		{"Malai Kebab", "Creamy cashew-marinated chicken kebabs", 300, false, 1, 510, "Kebab"},
		{"Rumali Roti (2 pc)", "Handkerchief-thin rotis off the tawa", 60, true, 0, 200, "Bread"},
		{"Mutton Galouti", "Melting galouti patties with ulte tawa paratha", 340, false, 1, 490, "Kebab"},
	},
	"seafood": {
		{"Mangalorean Fish Curry", "Seer fish in roasted coconut masala", 380, false, 2, 460, "Main Course"},
		{"Neer Dosa (3 pc)", "Lacy rice crepes, perfect with fish curry", 120, true, 0, 260, "Bread"},
		{"Kane Rava Fry", "Ladyfish semolina fry, Mangalore style", 340, false, 2, 420, "Starter"},
		{"Squid Sukka", "Dry-roasted squid with Kundapur masala", 360, false, 3, 380, "Starter"},
		{"Prawn Ghee Roast", "Ghee-laden spicy prawns, coastal legend", 400, false, 3, 440, "Starter"},
		{"Fish Thali", "Fish curry, fry, rice and papad — full coastal meal", 300, false, 2, 680, "Thali"},
	},
}

// Corpus is the immutable generated dataset.
type Corpus struct {
	Restaurants []Restaurant
	Items       []MenuItem
	byRest      map[string][]MenuItem
	byItem      map[string]MenuItem
	byName      map[string]Restaurant
}

var (
	once     sync.Once
	corpus   *Corpus
	corpusMu sync.RWMutex
)

// Get builds (once) and returns the corpus. Deterministic: same output every run.
func Get() *Corpus {
	once.Do(func() {
		rng := rand.New(rand.NewSource(20260928))
		c := &Corpus{
			byRest: map[string][]MenuItem{},
			byItem: map[string]MenuItem{},
			byName: map[string]Restaurant{},
		}
		for i, sr := range seedRestaurants {
			area := Areas[sr.area]
			r := Restaurant{
				ID:          fmt.Sprintf("r%02d", i+1),
				Name:        sr.name,
				Cuisines:    sr.cuisines,
				Area:        sr.area,
				Rating:      sr.rating,
				Ratings:     400 + rng.Intn(4600),
				PriceForTwo: sr.price,
				VegOnly:     sr.vegOnly,
				PrepMin:     12 + rng.Intn(14),
				Lat:         area.Lat + (rng.Float64()-0.5)*0.008,
				Lng:         area.Lng + (rng.Float64()-0.5)*0.008,
			}
			c.Restaurants = append(c.Restaurants, r)
			c.byName[r.Name] = r

			seen := map[string]bool{}
			var addBank = func(key string) {
				for _, bi := range banks[key] {
					if seen[bi.name] {
						continue
					}
					seen[bi.name] = true
					if r.VegOnly && !bi.veg {
						continue
					}
					// deterministic ±15% price jitter, rounded to nearest 10
					p := float64(bi.price) * (0.85 + rng.Float64()*0.30)
					price := int(math.Round(p/10.0) * 10)
					it := MenuItem{
						ID:          fmt.Sprintf("%s-i%02d", r.ID, len(c.byRest[r.ID])+1),
						RestID:      r.ID,
						RestName:    r.Name,
						Name:        bi.name,
						Description: bi.desc,
						Category:    bi.cat,
						Cuisine:     key,
						Area:        r.Area,
						Price:       price,
						Veg:         bi.veg,
						Spice:       bi.spice,
						Calories:    bi.cal + rng.Intn(60),
						Rating:      math.Round((3.4+rng.Float64()*1.5)*10) / 10,
						Bestseller:  rng.Float64() < 0.25,
					}
					if it.Rating > 5 {
						it.Rating = 4.9
					}
					c.Items = append(c.Items, it)
					c.byRest[r.ID] = append(c.byRest[r.ID], it)
					c.byItem[it.ID] = it
				}
			}
			for _, key := range sr.cuisines {
				addBank(key)
			}
			// every restaurant gets a drink and a sweet so the agent can
			// round out a meal anywhere
			addBank("beverages")
			if !hasCategory(c.byRest[r.ID], "Dessert") {
				addBank("desserts")
			}
		}
		corpus = c
	})
	return corpus
}

func hasCategory(items []MenuItem, cat string) bool {
	for _, it := range items {
		if it.Category == cat {
			return true
		}
	}
	return false
}

// RestByID returns the restaurant and whether it exists.
func (c *Corpus) RestByID(id string) (Restaurant, bool) {
	for _, r := range c.Restaurants {
		if r.ID == id {
			return r, true
		}
	}
	return Restaurant{}, false
}

// Menu returns the full menu for a restaurant in stable order.
func (c *Corpus) Menu(restID string) []MenuItem {
	return c.byRest[restID]
}

// ItemByID returns the menu item and whether it exists.
func (c *Corpus) ItemByID(id string) (MenuItem, bool) {
	it, ok := c.byItem[id]
	return it, ok
}

// SearchRestaurants filters and ranks restaurants. All filters optional.
// Ranking: rating desc, then cheaper price_for_two.
func (c *Corpus) SearchRestaurants(cuisine, area string, maxPriceForTwo, minRating int, vegOnly bool) []Restaurant {
	var out []Restaurant
	for _, r := range c.Restaurants {
		if cuisine != "" && !containsFold(joinLower(r.Cuisines), strings.ToLower(cuisine)) {
			continue
		}
		if area != "" && !strings.EqualFold(strings.TrimSpace(area), r.Area) &&
			!containsFold(r.Area, strings.ToLower(strings.TrimSpace(area))) {
			continue
		}
		if maxPriceForTwo > 0 && r.PriceForTwo > maxPriceForTwo {
			continue
		}
		if minRating > 0 && r.Rating < float64(minRating) {
			continue
		}
		if vegOnly && !r.VegOnly {
			continue
		}
		out = append(out, r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Rating != out[j].Rating {
			return out[i].Rating > out[j].Rating
		}
		return out[i].PriceForTwo < out[j].PriceForTwo
	})
	return out
}

// SearchDishes does a keyword match over item name + description + category
// and ranks by keyword hits then item rating.
func (c *Corpus) SearchDishes(query string, maxPrice int, vegOnly bool, minRating float64) []MenuItem {
	q := strings.ToLower(strings.TrimSpace(query))
	terms := strings.Fields(q)
	if len(terms) == 0 {
		return nil
	}
	type scored struct {
		it MenuItem
		s  int
	}
	var out []scored
	for _, it := range c.Items {
		hay := strings.ToLower(it.Name + " " + it.Description + " " + it.Category + " " + it.RestName)
		s := 0
		for _, t := range terms {
			if strings.Contains(hay, t) {
				s++
			}
		}
		if s == 0 {
			continue
		}
		if maxPrice > 0 && it.Price > maxPrice {
			continue
		}
		if vegOnly && !it.Veg {
			continue
		}
		if minRating > 0 && it.Rating < minRating {
			continue
		}
		out = append(out, scored{it, s})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].s != out[j].s {
			return out[i].s > out[j].s
		}
		return out[i].it.Rating > out[j].it.Rating
	})
	items := make([]MenuItem, 0, len(out))
	for _, o := range out {
		items = append(items, o.it)
	}
	return items
}

func joinLower(ss []string) string {
	var b strings.Builder
	for _, s := range ss {
		b.WriteString(strings.ToLower(s))
		b.WriteByte(' ')
	}
	return b.String()
}

func containsFold(hay, needle string) bool {
	return strings.Contains(strings.ToLower(hay), strings.ToLower(needle))
}

// HaversineKm returns the great-circle distance between two coordinates.
func HaversineKm(lat1, lng1, lat2, lng2 float64) float64 {
	const R = 6371.0
	dLat := (lat2 - lat1) * math.Pi / 180
	dLng := (lng2 - lng1) * math.Pi / 180
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1*math.Pi/180)*math.Cos(lat2*math.Pi/180)*
			math.Sin(dLng/2)*math.Sin(dLng/2)
	return R * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}
