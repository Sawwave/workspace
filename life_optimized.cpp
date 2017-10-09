//I agree to share name on timing data
//Tim Anderson
#include "../Clock.hpp"
#include <iostream>
#include <cstdint>

/*Conway's game of life, optimized


relies on keeping a pipeline of sums of the columns of the neighbors
Cells are stored as nybbles in a uint64_t so that 16 cells can be computed
at the same time, with no branches or cmp instructions required.

by adding up the surrounding cells, resulting in values 0x0 through 0x8,
ORing with 1 for alive cells and XORing with 0xC results in 0xFF for all
cells that will be alive next repetition. two bitwise ANDS collapse this state
to a 0 or 1 boolean state.

The most recently computed state is stored in bit 0, and the previous state is in
bit 1. By bitshifting groups that have been computed this repetition right one when
they need to be checked, we never need to go through and bitshift the whole board
after computation
*/


constexpr uint32_t R = 1<<10;
constexpr uint32_t C = 1<<11;
constexpr uint8_t GROUP_BIT_SIZE = 64;
constexpr uint32_t CELLS_PER_GROUP = GROUP_BIT_SIZE /4;
constexpr uint32_t GROUPS_PER_ROW = (C / CELLS_PER_GROUP) + 1;
constexpr auto BOARD_SIZE_GROUPS = GROUPS_PER_ROW * (R+2);

constexpr uint64_t TOP_NYBBLET_BITTMASK = 0xCCCCCCCCCCCCCCCC;
constexpr uint64_t LOW_BIT_NYBBLE_BITMASK = 0x1111111111111111;

void runRow(uint64_t *board, uint32_t row);
void advance(uint64_t *board);


//helper prototypes
uint64_t getColSumsForGroup(uint64_t *board, size_t group_index, uint64_t &group_val_out);
void moveColSumsDownPipeline(uint64_t &prev_col_sum, uint64_t &curr_col_sums,
                             uint64_t &next_col_sums);
uint64_t createLeftNeighborsFromColSums(uint64_t curr_col_sums, uint64_t prev_col_sum);
uint64_t createRightNeighborsFromColSums(uint64_t curr_col_sums, uint64_t next_col_sums);
void updateBoardIndex(uint64_t *board, size_t board_index,
                      uint64_t new_state_bit_values, uint64_t prev_board_state);
uint64_t neighborSumToStateBit(uint64_t sum_neighbors, uint64_t group_value);

void advance(uint64_t *board){
  #pragma omp parallel for
    for(auto row = 1; row < R + 1; row++){
        runRow(board, row);
    }
}

void runRow(uint64_t *board, uint32_t row){
    //find the starting group locations for the top, mid, and bot rows
    auto row_start_index = row * GROUPS_PER_ROW;
    uint64_t next_group_vals= 0;
    uint64_t next_col_sums = getColSumsForGroup(board, row_start_index, next_group_vals);
    uint64_t curr_col_sums = 0;
    uint64_t prev_col_sum = 0;

    for(auto group = 1; group < GROUPS_PER_ROW; group++){
        moveColSumsDownPipeline(prev_col_sum, curr_col_sums, next_col_sums);
        //find the next values
        uint64_t curr_group_vals = next_group_vals;
        next_col_sums = getColSumsForGroup(board, row_start_index+group, next_group_vals);
        uint64_t left = createLeftNeighborsFromColSums(curr_col_sums, prev_col_sum);
        uint64_t right =  createRightNeighborsFromColSums(curr_col_sums, next_col_sums);
        uint64_t current_neighbors = curr_col_sums + left + right);
        uint64_t new_state = neighborSumToStateBit(current_neighbors, curr_group_vals);

        updateBoardIndex(board, row_start_index + group - 1, new_state, curr_group_vals);
    }
}

uint64_t getColSumsForGroup(uint64_t *board, size_t group_index, uint64_t &group_val_out){
    group_val_out = board[group_index] & LOW_BIT_NYBBLE_BITMASK;
    uint64_t colSums = group_val_out;
    uint64_t topvals = (board[group_index- GROUPS_PER_ROW] >> 1) & LOW_BIT_NYBBLE_BITMASK;
    uint64_t botvals = board[group_index + GROUPS_PER_ROW] & LOW_BIT_NYBBLE_BITMASK;
    colSums += topvals + botvals;
    return colSums;
}

uint64_t neighborSumToStateBit(uint64_t sum_neighbors, uint64_t group_value){
    sum_neighbors -= group_value;
    sum_neighbors |= group_value;
    sum_neighbors ^= TOP_NYBBLET_BITTMASK;
    sum_neighbors &= sum_neighbors >> 2;
    sum_neighbors &= sum_neighbors >>1;
    sum_neighbors &= LOW_BIT_NYBBLE_BITMASK;
    return sum_neighbors;
}

void moveColSumsDownPipeline(uint64_t &prev_col_sum, uint64_t &curr_col_sums,
                             uint64_t &next_col_sums){
    prev_col_sum = curr_col_sums;
    curr_col_sums = next_col_sums;
}

void updateBoardIndex(uint64_t *board, size_t board_index,
                      uint64_t new_state_bit_values, uint64_t prev_board_state){
    uint64_t prev_state_shifted = (prev_board_state & LOW_BIT_NYBBLE_BITMASK)<<1;
    prev_state_shifted |= new_state_bit_values & LOW_BIT_NYBBLE_BITMASK;
    board[board_index] = prev_state_shifted;
}


uint64_t createLeftNeighborsFromColSums(uint64_t curr_col_sums, uint64_t prev_col_sum){
    uint64_t left_neighbors = curr_col_sums >> 4;
    uint64_t prev_carry_cell = prev_col_sum << 60;

    uint64_t out = left_neighbors | prev_carry_cell;
    return left_neighbors | prev_carry_cell;
}

uint64_t createRightNeighborsFromColSums(uint64_t curr_col_sums, uint64_t next_col_sums){
    uint64_t right_neighbors = curr_col_sums << 4;
    uint64_t next_carry_cell = next_col_sums >> 60;
    uint64_t out = right_neighbors | next_carry_cell;
    return out;
}






/*  --------------------------------------------------------------------*/
//boilerplate code from original copy

void print_board(const uint64_t *board) {
    for(auto row=1; row<R+1; row++) {
        const auto rowStartIndex = row * GROUPS_PER_ROW;
        for (auto col=0; col<(C/CELLS_PER_GROUP); col++) {
            uint64_t groupValue = board[rowStartIndex + col];
            std::cout << std::setfill('0') << std::setw(16) <<
                      std::hex<<(groupValue& LOW_BIT_NYBBLE_BITMASK);
        }
        std::cout << std::endl;
    }
}


int main(int argc, char**argv) {
    if (argc == 3) {
        srand(0);
        unsigned int NUM_REPS = atoi(argv[1]);
        bool print = bool(atoi(argv[2]));

        uint64_t *board = (uint64_t*)calloc(BOARD_SIZE_GROUPS, sizeof(uint64_t));
        for(auto row=1; row<R+1; row++) {
            const auto rowStartIndex = row * GROUPS_PER_ROW;
            for (auto col=0; col < GROUPS_PER_ROW - 1; col++) {
                auto groupIndex = rowStartIndex + col;
                uint64_t groupValue = 0;
                for(auto cell = 0; cell < CELLS_PER_GROUP; cell++){
                    groupValue <<= 4;
                    groupValue |= (rand()%2) == 0;
                }
                board[groupIndex] = groupValue;
            }
        }
        if (print) {
            print_board(board);
        }
        Clock c;
        for (unsigned int rep=0; rep<NUM_REPS; ++rep) {
            advance(board);
        }

        c.ptock();
        if (print) {
            print_board(board);
        }

        else{
        }

    } else {
        std::cerr << "usage game-of-life <generations> <0:don't print, 1:print>"
                  << std::endl;
    }

    return 0;
}
