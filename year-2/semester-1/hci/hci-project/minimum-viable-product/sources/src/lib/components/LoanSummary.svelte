<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import Avatar from "$lib/components/Avatar.svelte";
    import MoneyIn from "$lib/components/icons/MoneyIn.svelte";
    import MoneyOut from "$lib/components/icons/MoneyOut.svelte";
    import { formatDateTime, formatMoney } from "$lib/formatting";
    import type { OverrideSvelteHTMLElement } from "$lib/utils.svelte";
    import { CalendarDays, User } from "lucide-svelte";

    let {
        createdOn,
        avatar,
        loanType,
        badge,
        nPayments,
        value,
        class: klass,
        ...props
    }: OverrideSvelteHTMLElement<
        "button",
        {
            createdOn: Date;
            avatar: {
                user: {
                    id: string;
                    displayName: string;
                    avatar: boolean;
                    avatarModified: Date;
                };
                isGroup: boolean;
            } | null;
            loanType: "byme" | "tome";
            badge: boolean;
            nPayments: number;
            value: number;
        }
    > = $props();

    let signedValue = $derived(loanType == "byme" ? value : -value);
</script>

<button
    {...props}
    class={[
        "flex h-16 items-center gap-2 overflow-hidden rounded-md font-semibold shadow-md",
        loanType == "byme" ? "bg-green-200 text-green-950" : "bg-red-200 text-red-950",
        klass
    ]}
>
    <Avatar class="ml-2" user={avatar?.user ?? null} isGroup={avatar?.isGroup} />
    <div class="flex flex-col gap-1">
        <div class="flex gap-1">
            <div class="flex items-center gap-1">
                <AppIcon icon={CalendarDays} strokeWidth="1.5" class="size-6" />
                <p class="leading-none">{formatDateTime(createdOn)}</p>
            </div>
            {#if loanType == "byme"}
                <div class="flex items-center gap-1">
                    <AppIcon icon={User} strokeWidth="1.5" class="size-6" />
                    <p class="leading-none">{nPayments}</p>
                </div>
            {/if}
        </div>
        <div class="flex items-center gap-1">
            <AppIcon
                icon={loanType == "byme" ? MoneyIn : MoneyOut}
                strokeWidth="1.5"
                class="size-6"
            />
            <p class="leading-none">
                {formatMoney(signedValue)}
            </p>
        </div>
    </div>
    {#if badge}
        <div class="ml-auto w-2 self-stretch bg-blue-500"></div>
    {/if}
</button>
