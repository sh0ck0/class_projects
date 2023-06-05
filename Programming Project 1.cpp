#include <iostream>
#include <iomanip>
#include<math.h>
using namespace std;
const int PEOPLE_PER_COKE_CASE = 6;
const int PEOPLE_PER_TABLE_CASE = 6;
const double PEOPLE_PER_WATER_CASE = 2;
const int CARD_PRICE = 200;
const int WATER_PRICE = 20;
const int COKE_PRICE = 30;
void menu();
int choice();
int number_of_guests();
int sweets(int);
int cards(int);
double tables(double);
int coke(int);
double water(double);
int card_prices(int, int);
int coke_price(int,int);
int water_price(int, int);
void display(int, int, int, int, int, int, int, int, int);
int main()
{
    cout << "Wedding Planning Assistant is pleased to be at your service!" << endl << endl;
    int user_input_choice, user_input_guests = 0, cards_amount = 0, sweets_amount = 0, tables_amount = 0, coke_cases = 0, water_cases = 0, card_cost = 0, coke_cost = 0 , water_cost = 0 ;
    while(true)
    {
        menu();
        user_input_choice = choice();
        if(user_input_choice == 1)
        {
            user_input_guests = number_of_guests();
        }
        else if(user_input_choice == 2)
        {
            cards_amount = cards(user_input_guests);
            sweets_amount = sweets(user_input_guests);
            if(cards_amount != 0 || sweets_amount != 0)
                cout << "You will need " << cards_amount << " cards and " << sweets_amount << " sweets." << endl << endl;
        }
        else if(user_input_choice == 3)
        {
            tables_amount = tables(user_input_guests);
            if (tables_amount != 0)
                cout << "You will need " << tables_amount << " tables." << endl << endl;
        }
        else if(user_input_choice == 4)
        {
            coke_cases = coke(user_input_guests);
            water_cases = water(user_input_guests);
            if(coke_cases != 0 || water_cases != 0)
                cout << "You will need " << coke_cases << "cases of coke and " << water_cases << " cases of water." << endl << endl;
        }
        else if(user_input_choice == 5)
        {
            card_cost = card_prices(user_input_guests, cards_amount);
            if(card_cost != 0)
                cout << "The cards will cost " << card_cost << " NOK." << endl << endl;
        }
        else if(user_input_choice == 6)
        {
            water_cost = water_price(user_input_guests, water_cases);
            coke_cost = coke_price(user_input_guests, coke_cases);
            if(water_cost != 0 || coke_cost != 0)
                cout << "the water cases will cost " << water_cost << " NOK and the coke cases will cost " << coke_cost << "NOK." << endl << endl;
        }
        else if(user_input_choice == 7)
        {
            display(user_input_guests, cards_amount, sweets_amount, tables_amount, coke_cases, water_cases, card_cost, coke_cost, water_cost);
        }
        else if(user_input_choice == 8)
        {
            break;
        }
    }
    return 0;

}
void menu()
{
    cout << "1. Enter number of invited guests. " << endl;
    cout << "2. Determine the number of invitation cards and sweets" << endl;
    cout << "3. Determine the number of tables needed" << endl;
    cout << "4. Determine drinks order " << endl;
    cout << "5. Cost of invitation cards " << endl;
    cout << "6. Cost of drinks " << endl;
    cout << "7. Display all information" << endl;
    cout << "8. Quit" << endl << endl;
}
int choice()
{
    int  i = 0;
    while(true)
    {
        cout << "Please enter a number for choosing category: ";
        cin >> i;
        cout << endl;
        if(1 <= i && i <= 8)
        {
            break;
        }
        else
            cout << "the input is not valid, Please try again." << endl << endl;
    }
    return i;
}
int number_of_guests()
{
    int i = 1;
    int guests;
    while (i > 0)
    {
        cout << "Enter the number of invited guests: ";
        cin >> guests;
        if (guests > 0)
            break;
        else
            cout << "The number of guests can't be less than 0. try again" << endl << endl;
    }
    return  guests;
}
int cards(int guests)
{
    int cards = 0;
    if(guests <= 0)
    {
        cout << "You have not completed option 1" << endl << endl;
        return 0;
    }
    else
    {
        cards = guests / 2;
        return cards;
    }
}
int sweets(int guests)
{
    int sweets = 0;
    if(guests <= 0)
    {
        return 0;
    }
    else
    {
        sweets = guests * 1.2;
        return sweets;
    }
}
double tables(double guests)
{
    if(guests <= 0)
    {
        cout << "You have not completed option 1" << endl << endl;
        return 0;
    }
    else
    {
            double tables;
            tables = guests / PEOPLE_PER_TABLE_CASE;
            if(ceil(tables) == floor(tables))
            {
                return tables;
            }
            else
            {
                int tables_round;
                tables_round = tables + 1;
                return tables_round;
            }
    }
}
int coke(int guests)
{
    if(guests <= 0)
    {
        cout << "You have not completed option 1" << endl << endl;
        return 0;
    }
    else
    {
        int coke_cases;
        coke_cases = guests / PEOPLE_PER_COKE_CASE;
        return coke_cases;
    }
}
double water(double guests)
{
    if(guests <= 0)
    {
        return 0;
    }
    else
    {
        double water_cases;
        water_cases = guests / PEOPLE_PER_WATER_CASE;
        return water_cases;
    }
}
int card_prices(int guests, int card_amount )
{
    if(guests <= 0 || card_amount <= 0)
    {
        cout << "You have not completed option 1 and/or option 2" << endl <<  endl;
        return 0;
    }
    else
    {
        int card_cost;
        card_cost = card_amount * CARD_PRICE;
        return card_cost;
    }

}
int water_price(int guests, int water_cases)
{
        if(guests <= 0 || water_cases <= 0)
    {
        cout << "You have not completed option 1 and/or option 4" << endl <<  endl;
        return 0;
    }
    else
    {

        int water_cost;
        water_cost = water_cases * WATER_PRICE;
        return water_cost;
    }
}
int coke_price(int guests, int coke_cases)
{
    if(guests <= 0 || coke_cases <= 0)
    {
        return 0;
    }
    else
    {

        int coke_amount;
        coke_amount = coke_cases * COKE_PRICE;
        return coke_amount;
    }
}
void display(int user_input_guests,  int cards_amount, int sweets_amount, int tables_amount, int coke_cases, int water_cases, int card_cost, int coke_cost, int water_cost)
{
    cout << "Needs: ";
    cout << cards_amount << " invitation cards, " << sweets_amount << " sweets, ";
    cout << tables_amount << " tables, ";
    cout << coke_cases << " cases of coke and " << water_cases << " cases of water." << endl;
    cout << "Cost of invitation cards: " << card_cost << " NOK" << endl;
    cout << "Cost of drinks: " << coke_cost << " NOK for coke and " << water_cost << " NOK for water with a total of " << water_cost + coke_cost << " NOK." << endl << endl;
}
