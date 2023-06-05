#include <iostream>
#include<string>
using namespace std;


int SearchForBook();
struct book
{
     string Title;
     string Author;
     int Code;
     int Year;
     int Delete;

};
struct delted_record
{
     string Title;
     string Author;
     int Code;
     int Year;
};
void menu();
void InsertRecord(book []);
void DeleteRecord(book [], delted_record []);
void UnDeleteRecord(book [], delted_record []);
void PrintBooks(book []);
void SearchForBook(book []);
int main()
{
    book book_arr[1000];
    delted_record delted_record_arr[1000];
    char choice;

    while (true)
    {
        menu();



        cout << "Please enter: ";
        cin >> choice;
        if(choice == 'A')
            InsertRecord(book_arr);
        else if(choice == 'D')
            DeleteRecord(book_arr, delted_record_arr);
        else if(choice == 'U')
            UnDeleteRecord(book_arr, delted_record_arr);
        else if(choice == 'P')
            PrintBooks(book_arr);
        else if (choice == 'S')
            SearchForBook(book_arr);
        else if(choice == 'Q')
            break;
        else
            cout << "not a vaild option, please enter again." << endl << endl;


    }
    return 0;
}
void menu()
{
    cout << "***********************" << endl;
    cout << "Please enter a command:" << endl;
    cout << "'A': Add an entry" << endl;
    cout << "'D': Delete an entry" << endl;
    cout << "'U': Undelete an entry" << endl;
    cout << "'P': Print the books" << endl;
    cout << "'S': Search in the list" << endl;
    cout << "'Q': Quit" << endl;
    cout << "***********************" << endl << endl;
}
void InsertRecord(book book_arr[])
{

    for(int i = 0; i < 1000; i++)
    {
        if(book_arr[i].Title == "")
        {
            cin.ignore();
            cout << endl;
            cout << "Please enter the name of the book: ";
            getline(cin, book_arr[i].Title);
            cout << endl;

            cout << "Please enter the Author of the book: ";
            getline(cin, book_arr[i].Author);
            cout << endl;

            cout << "Please enter the Code of the book: ";
            cin >> book_arr[i].Code;
            cout << endl;

            cout << "Please enter the publication year: ";
            cin >> book_arr[i].Year;
            cout << endl;

            break;

        }
        else
        {
            continue;
        }
    }



}
void DeleteRecord(book book_arr[], delted_record delted_record_arr[])
{

    for(int i = 0; i < 1000; i++)
    {
        if(book_arr[i].Delete == 0)

        {
            cout << "Please enter the code of the book you want to delete: ";
            cin >> book_arr[i].Delete;
            cout << endl;
            for(int n = 0; n < 1000; n++)
            {

                if(book_arr[i].Delete == book_arr[n].Code)
                {
                    cout << "The book " << book_arr[n].Title << " was deleted." << endl << endl;
                    for(int x = 0; x < 1000; x++)
                    {
                        if(delted_record_arr[x].Title == "")
                        {
                            delted_record_arr[x].Title = book_arr[n].Title;
                            delted_record_arr[x].Author = book_arr[n].Author;
                            delted_record_arr[x].Code = book_arr[n].Code;
                            delted_record_arr[x].Year = book_arr[n].Year;


                            book_arr[n].Title = "";
                            book_arr[n].Author = "";
                            book_arr[n].Code = 0;
                            book_arr[n].Year = 0;
                            break;

                        }
                        else
                            continue;
                    }
                }
                else
                    continue;
            break;

            }
        }
        else
        {
            cout << "The code was not in the system." << endl << endl;
            break;
        }
        break;
    }
}
void UnDeleteRecord(book book_arr[], delted_record delted_record_arr[])
{
    int code;
    cout << "Enter the code of the book you want to UnDelete: ";
    cin >> code;
    cout << endl;
    for(int i = 0; i < 1000; i++)
    {
        if(code == delted_record_arr[i].Code)
        {
            for(int index = 0; index < 1000; index++)
            {
                if(book_arr[index].Title == "")
                {
                    book_arr[index].Title = delted_record_arr[i].Title;
                    book_arr[index].Author = delted_record_arr[i].Author;
                    book_arr[index].Code = delted_record_arr[i].Code;
                    book_arr[index].Year = delted_record_arr[i].Year;

                    break;
                }
                else
                    continue;
            }
            cout << "The book: " <<  delted_record_arr[i].Title << ". was undeleted." << endl << endl;
            break;
        }
        else
        {
            cout << " The code: " << code << ". is not in the system." << endl << endl;

            break;
        }
    }
}
void PrintBooks(book book_arr[])
{
    for(int i = 0; i < 1000; i++)
    {
        if(book_arr[i].Title != "" )
        {
            cout << "Book " << i + 1 << "." << endl;
            cout <<"Title: " << book_arr[i].Title << endl;
            cout <<"Author: " << book_arr[i].Author << endl;
            cout <<"Code: " << book_arr[i].Code << endl;
            cout <<"publication year: " << book_arr[i].Year << endl;
            cout << endl << endl;
        }
        else
            break;
    }
}
void SearchForBook(book book_arr[])
{
    int Search;
    cout << "Please enter the code of the book you want to Search after: ";
    cin >> Search;
    cout << endl;
    for( int i = 0; i < 1000; i++)
    {
        if (Search == book_arr[i].Code)
        {
            cout << "The book with code " << Search << ": " <<  endl;
            cout <<"Title: " << book_arr[i].Title << endl;
            cout <<"Author: " << book_arr[i].Author << endl;
            cout <<"publication year: " << book_arr[i].Year << endl << endl;
            break;
        }
        else
        {
            continue;
        }
        cout << "The book with code " << Search << " is not in the system" << endl << endl;
        break;
    }
}

