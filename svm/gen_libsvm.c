#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define YEAR_NOW 2024
#define MONTH_NOW 1
#define DAY_NOW 1

int team_hash(char* str) {
    int sum = 0;
    for (int i = 0; i < 3; i++) {
        sum += str[i] * (i + 1);
    }
    sum %= 100;
    return sum;
}

int parse_bool(char* str) {
    if (!strcmp(str, "True")) return 1;
    else return 0;
}

int date_dif(int year, int month, int day) {
    struct tm date1 = { 0 }, date2 = { 0 };
    time_t time1, time2;

    date1.tm_year = year - 1900;
    date1.tm_mon = month - 1;
    date1.tm_mday = day;

    date2.tm_year = YEAR_NOW - 1900;
    date2.tm_mon = MONTH_NOW - 1;
    date2.tm_mday = DAY_NOW;

    time1 = mktime(&date1);
    time2 = mktime(&date2);

    double difference = difftime(time2, time1) / (60 * 60 * 24);
    return (int)difference;
}

int main() {
    FILE* train = fopen("../train_data.csv", "r");
    FILE* out = fopen("train_data_libsvm", "w");
    // FILE* list = fopen("libsvm_list", "w");

    char buffer[7000];

    // parse list
    fgets(buffer, sizeof(buffer), train);
    // char* name = buffer;
    // char* comma = strchr(name, ',');
    // int num = 1;
    // while (comma) {
    //     *comma = 0;
    //     fprintf(list, "%3.d %s\n", num++, name);
    //     name = comma + 1;
    //     comma = strchr(name, ',');
    // }
    // fclose(list);

    // parse data
    char* name, * comma;
    while (fgets(buffer, sizeof(buffer), train)) {
        char temp1[10], temp2[10];
        name = buffer;
        comma = strchr(name, ',');
        int num = 1;
        while (comma) {
            *comma = 0;

            if (num == 1) {
            } else if (num == 2) {
                fprintf(out, "home:%-2d ", team_hash(name));
            } else if (num == 3) {
                fprintf(out, "away:%-2d|", team_hash(name));
            } else if (num == 4) {
                char* year = name;
                char* month = strchr(year, '-');
                *month = 0;
                month++;
                char* day = strchr(month, '-');
                *day = 0;
                day++;
                sprintf(temp1, "1:%d", date_dif(atoi(year), atoi(month), atoi(day)));
            } else if (num == 5) {
                sprintf(temp2, "2:%d", parse_bool(name));
            } else if (num == 6) {
                fprintf(out, "%d %s %s ", parse_bool(name), temp1, temp2);
            } else if (num == 7 || num == 8) {
            } else {
                if (name != comma)
                    fprintf(out, "%d:%s ", num - 7, name);
            }
            name = comma + 1;
            comma = strchr(name, ',');
            num++;
        }
        comma = strchr(name, '\n');
        *comma = 0;
        if (name != comma)
            fprintf(out, "%d:%s ", num - 3, name);
        fprintf(out, "\n");
    }

    fclose(train);
    fclose(out);


    return 0;
}