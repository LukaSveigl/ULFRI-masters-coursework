<script lang="ts">
    import Avatar from "$lib/components/Avatar.svelte";
    import PopoverContentMoney from "$lib/components/PopoverContentMoney.svelte";
    import Popup from "$lib/components/Popup.svelte";
    import { onUpdateLoanPayments } from "$lib/components/PopupDetailsLoanByMe.telefunc";
    import { formatDateTime, formatMoney } from "$lib/formatting";
    import { Button } from "$lib/shadcn/components/ui/button";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import * as Popover from "$lib/shadcn/components/ui/popover";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import { mutation, sneakyState } from "$lib/utils.svelte";

    let {
        open = $bindable(false),
        loan: loanProp = null,
        onupdate
    }: {
        open?: boolean;
        loan?: {
            createdOn: Date;
            _group: {
                id: string;
                displayName: string;
                avatar: boolean;
                avatarModified: Date;
            } | null;
            _payments: {
                id: string;
                value: number;
                valuePaid: number;
                valuePaidConfirmed: number;
                _user: {
                    id: string;
                    displayName: string;
                    avatar: boolean;
                    avatarModified: Date;
                    username: string;
                } | null;
            }[];
        } | null;
        onupdate?: () => Promise<void>;
    } = $props();

    const mutLoan = mutation({
        get() {
            const proxied = $state($state.snapshot(loanProp));
            return proxied;
        },
        async set(loan) {
            if (loan == null) return false;
            await onUpdateLoanPayments(
                loan._payments.map((x) => ({
                    id: x.id,
                    valuePaid: x.valuePaid,
                    valuePaidConfirmed: x.valuePaidConfirmed
                }))
            );
            await onupdate?.();
        }
    });

    let loan = $derived(mutLoan.value);
</script>

<Popup bind:open>
    <Dialog.Header>
        <Dialog.Title>Loan Details</Dialog.Title>
    </Dialog.Header>
    {#if loan}
        <div class="flex flex-col items-stretch gap-1">
            <div class="flex items-center">
                <p>Issued</p>
                <p class="ml-auto">{formatDateTime(loan.createdOn)}</p>
            </div>
            {#if loan._group != null}
                <div class="flex items-center">
                    <p>To group</p>
                    <div class="ml-auto">
                        <div class="flex items-center gap-2">
                            <Avatar user={loan._group} isGroup size="small" />
                            <p class="leading-none">{loan._group.displayName}</p>
                        </div>
                    </div>
                </div>
            {/if}
            <div class="flex items-center">
                <p>Total</p>
                <p class="ml-auto">
                    {formatMoney(loan._payments.reduce((acc, x) => acc + x.value, 0))}
                </p>
            </div>
            <div class="flex items-center">
                <p>Total paid, confirmed</p>
                <p class="ml-auto">
                    {formatMoney(loan._payments.reduce((acc, x) => acc + x.valuePaidConfirmed, 0))}
                </p>
            </div>
            <div class="flex items-center">
                <p>Total paid, unconfirmed</p>
                <p class="ml-auto">
                    {formatMoney(loan._payments.reduce((acc, x) => acc + x.valuePaid, 0))}
                </p>
            </div>
        </div>
        <ScrollArea
            orientation="both"
            class="flex max-h-80 flex-col items-stretch rounded-md border-[1px] border-inherit"
        >
            <div class="grid auto-rows-auto grid-cols-[auto_auto_auto] gap-2">
                <div
                    class="col-span-full grid grid-cols-subgrid border-b border-inherit p-2 text-sm font-medium text-muted-foreground"
                >
                    <p>Value</p>
                    <p>Confirmed</p>
                    <p>Unconfirmed</p>
                </div>
                <ul class="col-span-3 grid auto-rows-auto grid-cols-subgrid gap-y-2">
                    {#each loan._payments as payment (payment.id)}
                        {@const valuePaidPopoverOpen = sneakyState(false)}
                        <li
                            class="col-span-3 grid grid-cols-subgrid grid-rows-[auto_auto] items-stretch justify-items-stretch gap-y-2 border-b border-inherit px-2 pb-2 last:border-b-0"
                        >
                            <div class="col-span-3 flex items-center justify-center gap-2">
                                <Avatar user={payment._user} size="small" />
                                <p class="leading-none">{payment._user?.displayName ?? "?"}</p>
                            </div>
                            <Button
                                class={["!px-2 shadow-md", mutLoan.applying && "animate-pulse"]}
                                variant={payment.valuePaidConfirmed == payment.value
                                    ? "default"
                                    : "outline"}
                                disabled={payment.valuePaidConfirmed == payment.value}
                                onclick={() => {
                                    payment.valuePaid = payment.value;
                                    payment.valuePaidConfirmed = payment.value;
                                    mutLoan.value = loan;
                                }}
                            >
                                {formatMoney(payment.value)}
                            </Button>
                            <Button
                                class={["!px-2 shadow-md", mutLoan.applying && "animate-pulse"]}
                                variant={payment.valuePaidConfirmed != 0 &&
                                payment.valuePaidConfirmed != payment.value
                                    ? "default"
                                    : "outline"}
                                disabled={payment.valuePaidConfirmed == payment.valuePaid}
                                onclick={() => {
                                    payment.valuePaidConfirmed = payment.valuePaid;
                                    mutLoan.value = loan;
                                }}
                            >
                                {formatMoney(payment.valuePaidConfirmed)}
                            </Button>
                            <Popover.Root bind:open={valuePaidPopoverOpen.value}>
                                <Popover.Trigger>
                                    {#snippet child({ props })}
                                        <Button
                                            {...props}
                                            class={[
                                                "!px-2 shadow-md",
                                                mutLoan.applying && "animate-pulse"
                                            ]}
                                            variant={payment.valuePaidConfirmed != payment.valuePaid
                                                ? "default"
                                                : "outline"}
                                        >
                                            {formatMoney(payment.valuePaid)}
                                        </Button>
                                    {/snippet}
                                </Popover.Trigger>
                                <Popover.Content collisionPadding={16}>
                                    <PopoverContentMoney
                                        startingValue={payment.valuePaid}
                                        maxValue={payment.value}
                                        onsubmit={(newValuePaid) => {
                                            payment.valuePaid = newValuePaid;
                                            if (newValuePaid < payment.valuePaidConfirmed) {
                                                payment.valuePaidConfirmed = newValuePaid;
                                            }
                                            mutLoan.value = loan;
                                            valuePaidPopoverOpen.value = false;
                                        }}
                                    />
                                </Popover.Content>
                            </Popover.Root>
                        </li>
                    {/each}
                </ul>
            </div>
        </ScrollArea>
    {/if}
</Popup>
